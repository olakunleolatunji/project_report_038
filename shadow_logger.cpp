// ============================================================================
// shadow_logger.cpp — Rung 1: shadow-mode attitude control logger
//
// Runs the full companion-computer control pipeline at 50 Hz with NO actuation
// authority. Every cycle:
//
//   1. burst-read MPU6050 over I2C
//   2. complementary filter -> roll/pitch estimate
//   3. read pilot setpoint from RC_CHANNELS
//   4. run outer PID -> desired angular rate  (COMPUTED, LOGGED, NOT SENT)
//   5. capture flight controller ATTITUDE (EKF) and ATTITUDE_TARGET (FC rates)
//   6. write one CSV row
//
// Nothing this program produces reaches the flight controller. The only MAVLink
// messages it transmits are SET_MESSAGE_INTERVAL requests at startup.
//
// Output CSV supports metrics 1-5:
//   1  loop period / jitter      -> dt_s, compute_us, overrun
//   2  CPU utilisation           -> cpu_busy_pct, proc_cpu_pct
//   3  SNR improvement           -> roll_acc vs roll_cf
//   4  CF vs EKF cross-validation-> roll_cf vs ekf_roll
//   5  command divergence        -> roll_cmd vs fc_rate_roll
//
// Raw accel/gyro are logged too, so the filter coefficient and the PID gains
// can be re-derived offline from the same flight without reflying.
//
// BUILD
//   git clone --depth 1 https://github.com/mavlink/c_library_v2.git ~/c_library_v2
//   g++ -O2 -Wall -o shadow_logger shadow_logger.cpp -I$HOME/c_library_v2
//   (if you installed libmavlink-dev instead, change the include below to
//    <mavlink/common/mavlink.h> and drop the -I flag)
//
// RUN
//   sudo ./shadow_logger <run_label> [output_dir]
//   e.g. sudo ./shadow_logger run1_bench_unloaded
//        sudo ./shadow_logger run4_hover_loaded /home/pi/logs
//
//   Produces  <dir>/<label>_<YYYYmmdd_HHMMSS>.csv        (per-cycle data)
//             <dir>/<label>_<YYYYmmdd_HHMMSS>.meta.txt   (conditions + summary)
//   The label is also the first column of every CSV row, so concatenated
//   logs stay distinguishable. Timestamped names never overwrite each other.
//   (sudo is only for SCHED_FIFO; it runs without it, with a warning)
//
// PREREQUISITE
//   MAVProxy must be forwarding to this port, e.g.
//     mavproxy.py --master=/dev/ttyAMA0 --baudrate=57600 \
//                 --out=udp:<laptop-ip>:14552 --out=udp:127.0.0.1:14553
//   Stop the Python script first — only one process can bind 14553.
// ============================================================================

#include <common/mavlink.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <netinet/in.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

// ============================================================================
// CONFIGURATION
// ============================================================================

static const double LOOP_HZ       = 50.0;
static const long   PERIOD_NS     = (long)(1e9 / LOOP_HZ);   // 20 ms
static const double DT_NOMINAL    = 1.0 / LOOP_HZ;

static const char*  I2C_DEV       = "/dev/i2c-1";
static const int    MPU_ADDR      = 0x68;
static const int    UDP_PORT      = 14553;

// Complementary filter coefficient (Section 3.5.2)
static const double ALPHA         = 0.98;

// Shadow PID gains. Set these to ArduPilot's own angle-loop P
// (ATC_ANG_RLL_P / ATC_ANG_PIT_P) for a like-for-like metric 5 comparison.
// ArduPilot's angle loop is effectively P-only, so Ki/Kd default to zero.
// These can also be changed offline — the log carries the raw error.
static const double KP_ROLL = 4.5, KI_ROLL = 0.0, KD_ROLL = 0.0;
static const double KP_PITCH= 4.5, KI_PITCH= 0.0, KD_PITCH= 0.0;

static const double I_MAX         = 50.0;    // integral clamp, deg/s
static const double RATE_MAX      = 200.0;   // output clamp, deg/s

// Pilot stick -> angle mapping
static const double MAX_ANGLE     = 30.0;
static const double PWM_CENTER    = 1500.0;
static const double PWM_RANGE     = 500.0;

// On ArduPilot, RC2 rising usually commands nose-DOWN. Verify on your own
// airframe by tilting the stick and watching pitch_sp vs ekf_pitch, then set
// this to +1 or -1 accordingly. A wrong sign silently corrupts metric 5.
static const double PITCH_SP_SIGN = -1.0;

// Our identity on the MAVLink network (do not collide with Mission Planner=255)
static const uint8_t MY_SYSID     = 254;
static const uint8_t MY_COMPID    = MAV_COMP_ID_ONBOARD_COMPUTER;   // 191

static const int    CALIB_SAMPLES = 400;

// ============================================================================
// GLOBALS
// ============================================================================

static volatile sig_atomic_t g_run = 1;
static void on_sigint(int) { g_run = 0; }

static int  g_i2c  = -1;
static int  g_sock = -1;
static struct sockaddr_in g_peer;     // where MAVProxy sends from
static bool g_peer_known = false;

// ============================================================================
// TIME
// ============================================================================

static inline double mono_now()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline void ts_add_ns(struct timespec* ts, long ns)
{
    ts->tv_nsec += ns;
    while (ts->tv_nsec >= 1000000000L) { ts->tv_nsec -= 1000000000L; ts->tv_sec++; }
}

// ============================================================================
// MPU6050
// ============================================================================

struct Imu {
    double ax, ay, az;      // g
    double gx, gy, gz;      // deg/s
};

static bool mpu_write(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = { reg, val };
    return write(g_i2c, buf, 2) == 2;
}

static bool mpu_init()
{
    g_i2c = open(I2C_DEV, O_RDWR);
    if (g_i2c < 0) { perror("open i2c"); return false; }
    if (ioctl(g_i2c, I2C_SLAVE, MPU_ADDR) < 0) { perror("ioctl I2C_SLAVE"); return false; }

    if (!mpu_write(0x6B, 0x00)) return false;   // wake
    usleep(50000);
    if (!mpu_write(0x1B, 0x08)) return false;   // gyro  +/-500 deg/s
    if (!mpu_write(0x1C, 0x00)) return false;   // accel +/-2 g
    if (!mpu_write(0x1A, 0x03)) return false;   // DLPF ~44 Hz, helps vs prop vibration
    usleep(50000);
    return true;
}

// Single 14-byte burst so accel and gyro come from the same sample instant.
static bool mpu_read(Imu* o)
{
    uint8_t reg = 0x3B;
    if (write(g_i2c, &reg, 1) != 1) return false;

    uint8_t d[14];
    if (read(g_i2c, d, 14) != 14) return false;

    auto w = [&](int hi) -> int16_t {
        return (int16_t)(((uint16_t)d[hi] << 8) | d[hi + 1]);
    };

    const double AS = 2.0   / 32768.0;
    const double GS = 500.0 / 32768.0;

    o->ax = w(0)  * AS;
    o->ay = w(2)  * AS;
    o->az = w(4)  * AS;
    // d[6],d[7] = temperature, skipped
    o->gx = w(8)  * GS;
    o->gy = w(10) * GS;
    o->gz = w(12) * GS;
    return true;
}

static inline double roll_from_accel(const Imu& m)
{
    return atan2(m.ay, m.az) * 180.0 / M_PI;
}

static inline double pitch_from_accel(const Imu& m)
{
    return atan2(-m.ax, sqrt(m.ay * m.ay + m.az * m.az)) * 180.0 / M_PI;
}

// ============================================================================
// MAVLINK OVER UDP
// ============================================================================

static bool udp_init()
{
    g_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (g_sock < 0) { perror("socket"); return false; }

    int flags = fcntl(g_sock, F_GETFL, 0);
    fcntl(g_sock, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in me;
    memset(&me, 0, sizeof(me));
    me.sin_family      = AF_INET;
    me.sin_addr.s_addr = htonl(INADDR_ANY);
    me.sin_port        = htons(UDP_PORT);

    if (bind(g_sock, (struct sockaddr*)&me, sizeof(me)) < 0) {
        perror("bind (is the Python script still running?)");
        return false;
    }
    return true;
}

static void mav_send(const mavlink_message_t* msg)
{
    if (!g_peer_known) return;
    uint8_t buf[MAVLINK_MAX_PACKET_LEN];
    uint16_t len = mavlink_msg_to_send_buffer(buf, msg);
    sendto(g_sock, buf, len, 0, (struct sockaddr*)&g_peer, sizeof(g_peer));
}

// Telemetry snapshot, refreshed as messages arrive
struct Telem {
    double roll_sp   = 0.0, pitch_sp   = 0.0;   // deg, from RC_CHANNELS
    double ekf_roll  = 0.0, ekf_pitch  = 0.0;   // deg, from ATTITUDE
    double fc_rate_roll = 0.0, fc_rate_pitch = 0.0; // deg/s, from ATTITUDE_TARGET
    double t_rc = 0.0, t_att = 0.0, t_tgt = 0.0;    // age tracking
    uint8_t  sysid = 0, compid = 0;
    bool     have_heartbeat = false;
    long     n_rc = 0, n_att = 0, n_tgt = 0;
};

static inline double pwm_to_angle(uint16_t pwm)
{
    if (pwm < 800 || pwm > 2200) return 0.0;     // invalid / unmapped channel
    double a = ((double)pwm - PWM_CENTER) / PWM_RANGE * MAX_ANGLE;
    if (a >  MAX_ANGLE) a =  MAX_ANGLE;
    if (a < -MAX_ANGLE) a = -MAX_ANGLE;
    return a;
}

// Drain the socket and dispatch by type. Never filter on a single message
// type here — doing so silently discards everything else.
static void mav_poll(Telem* t)
{
    uint8_t buf[2048];
    struct sockaddr_in from;
    socklen_t fromlen = sizeof(from);

    for (;;) {
        ssize_t n = recvfrom(g_sock, buf, sizeof(buf), 0,
                             (struct sockaddr*)&from, &fromlen);
        if (n <= 0) break;

        if (!g_peer_known) { g_peer = from; g_peer_known = true; }

        mavlink_message_t msg;
        mavlink_status_t  status;

        for (ssize_t i = 0; i < n; i++) {
            if (!mavlink_parse_char(MAVLINK_COMM_0, buf[i], &msg, &status))
                continue;

            double now = mono_now();

            switch (msg.msgid) {

            case MAVLINK_MSG_ID_HEARTBEAT:
                if (!t->have_heartbeat && msg.sysid != MY_SYSID) {
                    t->sysid  = msg.sysid;
                    t->compid = msg.compid;
                    t->have_heartbeat = true;
                }
                break;

            case MAVLINK_MSG_ID_RC_CHANNELS: {
                mavlink_rc_channels_t rc;
                mavlink_msg_rc_channels_decode(&msg, &rc);
                t->roll_sp  = pwm_to_angle(rc.chan1_raw);
                t->pitch_sp = PITCH_SP_SIGN * pwm_to_angle(rc.chan2_raw);
                t->t_rc = now;
                t->n_rc++;
                break;
            }

            case MAVLINK_MSG_ID_ATTITUDE: {
                mavlink_attitude_t a;
                mavlink_msg_attitude_decode(&msg, &a);
                t->ekf_roll  = a.roll  * 180.0 / M_PI;
                t->ekf_pitch = a.pitch * 180.0 / M_PI;
                t->t_att = now;
                t->n_att++;
                break;
            }

            case MAVLINK_MSG_ID_ATTITUDE_TARGET: {
                mavlink_attitude_target_t g;
                mavlink_msg_attitude_target_decode(&msg, &g);
                t->fc_rate_roll  = g.body_roll_rate  * 180.0 / M_PI;
                t->fc_rate_pitch = g.body_pitch_rate * 180.0 / M_PI;
                t->t_tgt = now;
                t->n_tgt++;
                break;
            }

            default: break;
            }
        }
    }
}

static void request_stream(const Telem& t, uint32_t msgid, double hz)
{
    mavlink_message_t msg;
    mavlink_msg_command_long_pack(
        MY_SYSID, MY_COMPID, &msg,
        t.sysid, t.compid,
        MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        (float)msgid,                 // param1: message id
        (float)(1e6 / hz),            // param2: interval, microseconds
        0, 0, 0, 0, 0);
    mav_send(&msg);
}

// ============================================================================
// CPU SAMPLING  (metric 2)
// ============================================================================

struct CpuSampler {
    unsigned long long prev_total = 0, prev_idle = 0;
    unsigned long long prev_proc  = 0;
    double last_sample_t = 0.0;
    double busy_pct = 0.0;      // whole-system
    double proc_pct = 0.0;      // this process

    void read_now(double now)
    {
        // ---- /proc/stat : whole system ----
        FILE* f = fopen("/proc/stat", "r");
        if (f) {
            unsigned long long u, n, s, i, io, irq, sirq, st;
            if (fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu",
                       &u, &n, &s, &i, &io, &irq, &sirq, &st) == 8) {
                unsigned long long idle  = i + io;
                unsigned long long total = u + n + s + idle + irq + sirq + st;
                if (prev_total && total > prev_total) {
                    double dt_total = (double)(total - prev_total);
                    double dt_idle  = (double)(idle  - prev_idle);
                    busy_pct = 100.0 * (dt_total - dt_idle) / dt_total;
                }
                prev_total = total;
                prev_idle  = idle;
            }
            fclose(f);
        }

        // ---- /proc/self/stat : this process (fields 14 utime, 15 stime) ----
        f = fopen("/proc/self/stat", "r");
        if (f) {
            char line[1024];
            if (fgets(line, sizeof(line), f)) {
                char* p = strrchr(line, ')');
                if (p) {
                    int field = 2;
                    unsigned long long ut = 0, stm = 0;
                    char* tok = strtok(p + 2, " ");
                    while (tok) {
                        field++;
                        if (field == 14) ut  = strtoull(tok, NULL, 10);
                        if (field == 15) { stm = strtoull(tok, NULL, 10); break; }
                        tok = strtok(NULL, " ");
                    }
                    unsigned long long proc = ut + stm;
                    double hz = (double)sysconf(_SC_CLK_TCK);
                    double dt = now - last_sample_t;
                    if (prev_proc && dt > 0.0)
                        proc_pct = 100.0 * ((double)(proc - prev_proc) / hz) / dt;
                    prev_proc = proc;
                }
            }
            fclose(f);
        }

        last_sample_t = now;
    }
};

// ============================================================================
// PID  (shadow — output is logged, never transmitted)
// ============================================================================

struct Pid {
    double kp, ki, kd;
    double integral   = 0.0;
    double prev_meas  = 0.0;
    bool   first      = true;

    double p_term = 0.0, i_term = 0.0, d_term = 0.0;

    double step(double setpoint, double measured, double dt)
    {
        double err = setpoint - measured;

        p_term = kp * err;

        integral += err * dt;
        if (integral >  I_MAX / (ki > 0 ? ki : 1.0)) integral =  I_MAX / (ki > 0 ? ki : 1.0);
        if (integral < -I_MAX / (ki > 0 ? ki : 1.0)) integral = -I_MAX / (ki > 0 ? ki : 1.0);
        i_term = ki * integral;

        // derivative on measurement, not on error: avoids a setpoint-step kick
        if (first) { prev_meas = measured; first = false; }
        d_term = -kd * (measured - prev_meas) / (dt > 0 ? dt : DT_NOMINAL);
        prev_meas = measured;

        double out = p_term + i_term + d_term;
        if (out >  RATE_MAX) out =  RATE_MAX;
        if (out < -RATE_MAX) out = -RATE_MAX;
        return out;
    }
};

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv)
{
    // ------------------------------------------------------------------
    // Run identity. The label drives the filename, a sidecar .meta.txt,
    // and a CSV column — so a log can never be mistaken for another run's.
    //   ./shadow_logger run1_bench_unloaded
    //   ./shadow_logger run4_hover_loaded
    // ------------------------------------------------------------------
    std::string label = (argc > 1) ? argv[1] : "unlabelled";
    std::string dir   = (argc > 2) ? argv[2] : "/dev/shm";

    char stamp[32];
    time_t wall = time(NULL);
    strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", localtime(&wall));

    std::string base      = dir + "/" + label + "_" + stamp;
    std::string csv_path  = base + ".csv";
    std::string meta_path = base + ".meta.txt";
    const char* out_path  = csv_path.c_str();

    signal(SIGINT,  on_sigint);
    signal(SIGTERM, on_sigint);

    // ---- real-time priority (Section 3.4.4) ----
    bool have_fifo = false;
    struct sched_param sp;
    sp.sched_priority = 80;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) == 0) {
        have_fifo = true;
        printf("SCHED_FIFO priority 80 acquired\n");
    } else {
        printf("WARNING: SCHED_FIFO unavailable (run with sudo). "
               "Timing figures will include scheduler preemption.\n");
    }

    if (!mpu_init()) { fprintf(stderr, "MPU6050 init failed\n"); return 1; }
    if (!udp_init()) { fprintf(stderr, "UDP init failed\n");     return 1; }

    // ---- wait for the vehicle ----
    printf("Waiting for heartbeat on UDP %d ...\n", UDP_PORT);
    Telem tel;
    while (g_run && !tel.have_heartbeat) { mav_poll(&tel); usleep(20000); }
    if (!g_run) return 0;
    printf("Connected — vehicle sysid %u, compid %u\n", tel.sysid, tel.compid);

    // ---- ask for the streams we need at 50 Hz ----
    // ATTITUDE_TARGET in particular is often absent from the default stream.
    for (int i = 0; i < 3; i++) {
        request_stream(tel, MAVLINK_MSG_ID_ATTITUDE,        50.0);
        request_stream(tel, MAVLINK_MSG_ID_ATTITUDE_TARGET, 50.0);
        request_stream(tel, MAVLINK_MSG_ID_RC_CHANNELS,     50.0);
        usleep(100000);
        mav_poll(&tel);
    }

    // ---- calibration ----
    printf("Hold the airframe flat and still — calibrating (%d samples)...\n",
           CALIB_SAMPLES);
    sleep(2);

    double gx_off = 0, gy_off = 0, gz_off = 0, roll_off = 0, pitch_off = 0;
    for (int i = 0; i < CALIB_SAMPLES; i++) {
        Imu m;
        if (!mpu_read(&m)) { fprintf(stderr, "IMU read failed during calib\n"); return 1; }
        gx_off    += m.gx;
        gy_off    += m.gy;
        gz_off    += m.gz;
        roll_off  += roll_from_accel(m);
        pitch_off += pitch_from_accel(m);
        usleep(5000);
    }
    gx_off /= CALIB_SAMPLES;  gy_off /= CALIB_SAMPLES;  gz_off /= CALIB_SAMPLES;
    roll_off /= CALIB_SAMPLES; pitch_off /= CALIB_SAMPLES;

    printf("Offsets — roll %.2f deg, pitch %.2f deg, gx %.3f, gy %.3f, gz %.3f\n",
           roll_off, pitch_off, gx_off, gy_off, gz_off);

    // ---- sidecar metadata: what this run actually was ----
    {
        FILE* mf = fopen(meta_path.c_str(), "w");
        if (mf) {
            char host[256] = {0};
            gethostname(host, sizeof(host) - 1);
            char walltime[64];
            time_t w = time(NULL);
            strftime(walltime, sizeof(walltime), "%Y-%m-%d %H:%M:%S %Z", localtime(&w));

            fprintf(mf, "run_label     : %s\n", label.c_str());
            fprintf(mf, "started       : %s\n", walltime);
            fprintf(mf, "host          : %s\n", host);
            fprintf(mf, "csv           : %s\n", csv_path.c_str());
            fprintf(mf, "mode          : SHADOW (no command transmitted)\n");
            fprintf(mf, "loop_hz       : %.1f\n", LOOP_HZ);
            fprintf(mf, "sched_fifo    : %s\n", have_fifo ? "yes (prio 80)" : "NO");
            fprintf(mf, "alpha         : %.4f\n", ALPHA);
            fprintf(mf, "gains_roll    : Kp=%.4f Ki=%.4f Kd=%.4f\n", KP_ROLL, KI_ROLL, KD_ROLL);
            fprintf(mf, "gains_pitch   : Kp=%.4f Ki=%.4f Kd=%.4f\n", KP_PITCH, KI_PITCH, KD_PITCH);
            fprintf(mf, "pitch_sp_sign : %+.0f\n", PITCH_SP_SIGN);
            fprintf(mf, "vehicle       : sysid %u compid %u\n", tel.sysid, tel.compid);
            fprintf(mf, "calib_offsets : roll %.3f deg, pitch %.3f deg, "
                        "gx %.4f gy %.4f gz %.4f dps\n",
                    roll_off, pitch_off, gx_off, gy_off, gz_off);

            // Honest record of what else was on the CPU during this run.
            fprintf(mf, "\ntop processes at start (ps -eo comm,pcpu --sort=-pcpu):\n");
            FILE* ps = popen("ps -eo comm,pcpu --sort=-pcpu 2>/dev/null | head -12", "r");
            if (ps) {
                char line[256];
                while (fgets(line, sizeof(line), ps)) fprintf(mf, "  %s", line);
                pclose(ps);
            }
            fprintf(mf, "\nnotes (fill in by hand):\n"
                        "  motors        : off / spinning / hover\n"
                        "  comms stack   : stopped / running\n"
                        "  observations  : \n");
            fclose(mf);
            printf("Metadata: %s\n", meta_path.c_str());
        }
    }

    // ---- open log ----
    FILE* log = fopen(out_path, "w");
    if (!log) { perror("fopen log"); return 1; }
    static char logbuf[1 << 20];
    setvbuf(log, logbuf, _IOFBF, sizeof(logbuf));

    fprintf(log,
        "run_label,seq,t_s,dt_s,compute_us,overrun,"
        "ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,"
        "roll_acc_deg,pitch_acc_deg,roll_cf_deg,pitch_cf_deg,"
        "roll_sp_deg,pitch_sp_deg,roll_err_deg,pitch_err_deg,"
        "roll_cmd_dps,pitch_cmd_dps,roll_p,roll_i,roll_d,"
        "ekf_roll_deg,ekf_pitch_deg,fc_rate_roll_dps,fc_rate_pitch_dps,"
        "age_att_s,age_tgt_s,age_rc_s,cpu_busy_pct,proc_cpu_pct\n");

    // ---- state ----
    double roll_cf = 0.0, pitch_cf = 0.0;
    Pid pid_roll  { KP_ROLL,  KI_ROLL,  KD_ROLL  };
    Pid pid_pitch { KP_PITCH, KI_PITCH, KD_PITCH };
    CpuSampler cpu;

    double t0 = mono_now();
    cpu.read_now(t0);

    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    double prev_wake = mono_now();

    long   seq = 0;
    double cpu_next = t0 + 1.0;

    printf("\nLogging to %s — Ctrl-C to stop.\n", out_path);
    printf("SHADOW MODE: no command is transmitted to the flight controller.\n\n");

    while (g_run) {

        // ---- wait for the next 20 ms deadline ----
        ts_add_ns(&next, PERIOD_NS);
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);

        double wake = mono_now();
        double dt   = wake - prev_wake;
        prev_wake   = wake;
        if (dt <= 0.0)  dt = DT_NOMINAL;
        if (dt >  0.5)  dt = 0.5;

        double c0 = mono_now();

        // ---- 1. sensors ----
        Imu m;
        if (!mpu_read(&m)) { fprintf(stderr, "IMU read failed at seq %ld\n", seq); continue; }
        m.gx -= gx_off;  m.gy -= gy_off;  m.gz -= gz_off;

        double roll_acc  = roll_from_accel(m)  - roll_off;
        double pitch_acc = pitch_from_accel(m) - pitch_off;

        // ---- 2. complementary filter ----
        roll_cf  = ALPHA * (roll_cf  + m.gx * dt) + (1.0 - ALPHA) * roll_acc;
        pitch_cf = ALPHA * (pitch_cf + m.gy * dt) + (1.0 - ALPHA) * pitch_acc;

        // ---- 3. telemetry ----
        mav_poll(&tel);

        // ---- 4. shadow PID (result is logged, not sent) ----
        double roll_err  = tel.roll_sp  - roll_cf;
        double pitch_err = tel.pitch_sp - pitch_cf;
        double roll_cmd  = pid_roll.step (tel.roll_sp,  roll_cf,  dt);
        double pitch_cmd = pid_pitch.step(tel.pitch_sp, pitch_cf, dt);

        // ---- 5. CPU sample at 1 Hz ----
        if (wake >= cpu_next) { cpu.read_now(wake); cpu_next = wake + 1.0; }

        double compute_us = (mono_now() - c0) * 1e6;
        int    overrun    = (dt > DT_NOMINAL * 1.5) ? 1 : 0;

        // ---- 6. log ----
        fprintf(log,
            "%s,%ld,%.6f,%.6f,%.1f,%d,"
            "%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,"
            "%.4f,%.4f,%.4f,%.4f,"
            "%.4f,%.4f,%.4f,%.4f,"
            "%.4f,%.4f,%.4f,%.4f,%.4f,"
            "%.4f,%.4f,%.4f,%.4f,"
            "%.3f,%.3f,%.3f,%.2f,%.2f\n",
            label.c_str(), seq, wake - t0, dt, compute_us, overrun,
            m.ax, m.ay, m.az, m.gx, m.gy, m.gz,
            roll_acc, pitch_acc, roll_cf, pitch_cf,
            tel.roll_sp, tel.pitch_sp, roll_err, pitch_err,
            roll_cmd, pitch_cmd, pid_roll.p_term, pid_roll.i_term, pid_roll.d_term,
            tel.ekf_roll, tel.ekf_pitch, tel.fc_rate_roll, tel.fc_rate_pitch,
            tel.t_att > 0 ? wake - tel.t_att : -1.0,
            tel.t_tgt > 0 ? wake - tel.t_tgt : -1.0,
            tel.t_rc  > 0 ? wake - tel.t_rc  : -1.0,
            cpu.busy_pct, cpu.proc_pct);

        // ---- console, 5 Hz ----
        if (seq % 10 == 0) {
            printf("\rsp %+6.1f/%+6.1f  cf %+6.1f/%+6.1f  ekf %+6.1f/%+6.1f  "
                   "cmd %+6.1f  fc %+6.1f  dt %5.2fms  cpu %4.1f%%   ",
                   tel.roll_sp, tel.pitch_sp, roll_cf, pitch_cf,
                   tel.ekf_roll, tel.ekf_pitch, roll_cmd, tel.fc_rate_roll,
                   dt * 1000.0, cpu.busy_pct);
            fflush(stdout);
        }

        seq++;
    }

    // ---- shutdown ----
    fflush(log);
    fclose(log);
    if (g_i2c  >= 0) close(g_i2c);
    if (g_sock >= 0) close(g_sock);

    double elapsed = mono_now() - t0;

    // Append the summary to the sidecar so the record lives with the run.
    {
        FILE* mf = fopen(meta_path.c_str(), "a");
        if (mf) {
            fprintf(mf, "\nsummary:\n");
            fprintf(mf, "  duration_s    : %.1f\n", elapsed);
            fprintf(mf, "  cycles        : %ld\n", seq);
            fprintf(mf, "  effective_hz  : %.2f\n", elapsed > 0 ? seq / elapsed : 0.0);
            fprintf(mf, "  msgs ATTITUDE : %ld\n", tel.n_att);
            fprintf(mf, "  msgs ATT_TGT  : %ld\n", tel.n_tgt);
            fprintf(mf, "  msgs RC_CHAN  : %ld\n", tel.n_rc);
            fclose(mf);
        }
    }

    printf("\n\nStopped.\n");
    printf("  cycles          : %ld over %.1f s  (%.2f Hz effective)\n",
           seq, elapsed, elapsed > 0 ? seq / elapsed : 0.0);
    printf("  messages seen   : ATTITUDE %ld, ATTITUDE_TARGET %ld, RC_CHANNELS %ld\n",
           tel.n_att, tel.n_tgt, tel.n_rc);
    if (tel.n_tgt == 0)
        printf("  WARNING: no ATTITUDE_TARGET received — metric 5 has no reference.\n"
               "           Raise SR2_EXTRA1 / check the message interval request.\n");
    if (tel.n_rc == 0)
        printf("  WARNING: no RC_CHANNELS received — setpoints were all zero.\n");
    printf("  log             : %s\n", out_path);
    return 0;
}
