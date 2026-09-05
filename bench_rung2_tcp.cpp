// ============================================================================
// bench_rung2_tcp.cpp — Rung 2: single-axis bench-rig closed-loop controller
//                       (TCP transport variant of bench_rung2.cpp)
//
// Same measurement pipeline, PID, watchdogs, and CSV format as the UDP
// version. The only difference is transport: this program connects to a
// mavlink-router TcpEndpoint as a TCP client instead of doing UDP send/recv.
//
// WHY TCP
//   With the SSH port forward
//       ssh -L 5760:127.0.0.1:5760 pi-drone@pi-host.local
//   Mission Planner on the laptop reaches the Pi's mavlink-router at
//   localhost:5760 through the tunnel. If mavlink-router serves a single
//   TcpEndpoint on 5760, both Mission Planner (via the tunnel) and this
//   program (natively on the Pi) can connect as TCP clients and share the
//   same FC stream.
//
//   TCP also removes the "silent Server endpoint" quirk that made the UDP
//   version have to keep pumping heartbeats to be noticed — a TCP client is
//   visible to the router the moment connect() returns.
//
// MAVLINK-ROUTER CONFIG
//   Add a TcpEndpoint that listens on 5760 (or whichever port you tunnel):
//
//     [TcpEndpoint bench]
//     Mode = Server
//     Address = 127.0.0.1
//     Port = 5760
//
//   Then reload mavlink-router:  sudo systemctl restart mavlink-router
//
//   Confirm it is listening:
//     ss -ltnp | grep 5760
//
//   If mavlink-router's build only accepts one TCP client per TcpEndpoint,
//   add a second TcpEndpoint on a different port for Mission Planner and
//   keep 5760 for this program.
//
// SAFETY ENVELOPE — unchanged from the UDP version:
//   - refuses to run unless FC reports mode GUIDED_NOGPS (mode 20 on Copter)
//   - refuses to run unless user confirms rig readiness by typing "READY"
//   - watchdog disarms if FC ATTITUDE stream stops for > 500 ms
//   - watchdog disarms if IMU read fails
//   - watchdog disarms if PID output saturates for > 1 s continuously
//   - watchdog disarms if the TCP peer closes on us
//   - SIGINT / SIGTERM / SIGPIPE / any exception path calls disarm-then-exit
//   - rate command clamped to RATE_MAX_CMD (60 dps default)
//
// BUILD
//   g++ -O2 -Wall -o bench_rung2_tcp bench_rung2_tcp.cpp -I$HOME/c_library_v2
//
// RUN
//   sudo ./bench_rung2_tcp <label> <axis> <throttle> [endpoint]
//     axis:     roll | pitch | yaw
//     throttle: 0.00 to 0.60
//     endpoint: defaults to tcp:127.0.0.1:5760
//               tcp:HOST:PORT   connect to a mavlink-router TcpEndpoint
//
//   Example flow:
//     sudo ./bench_rung2_tcp stageA_verify roll 0.05
//     sudo ./bench_rung2_tcp stageB_arm    roll 0.00
//     sudo ./bench_rung2_tcp stageC_thr    roll 0.20
//     sudo ./bench_rung2_tcp stageD_ctrl   roll 0.25
//
// STOPPING
//   Ctrl-C at any time: sends DISARM (with retries), then exits.
//   Mission Planner disarm button also works — the FC just accepts it.
//   The FC will also disarm itself if the setpoint stream stops for > 3 s
//   (GUID_TIMEOUT), or if the GCS heartbeat stops (FS_GCS_ENABLE).
// ============================================================================

#include <common/mavlink.h>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
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
static const long   PERIOD_NS     = (long)(1e9 / LOOP_HZ);
static const double DT_NOMINAL    = 1.0 / LOOP_HZ;

static const char*  I2C_DEV       = "/dev/i2c-1";
static const int    MPU_ADDR      = 0x68;
static const int    TCP_PORT      = 5760;

// TCP client toward a mavlink-router TcpEndpoint. See the header block for
// the matching router configuration.
static const char*  DEFAULT_ENDPOINT = "tcp:127.0.0.1:5760";

static const double ALPHA         = 0.98;
// KP_ROLL reduced 4.5 -> 1.0 for the first props-on run: the MPU6050 estimate
// picks up ~3.6 deg of vibration once the motors spool, and at 4.5 that became
// +/-9.6 dps of rate command derived purely from noise. At 1.0 the same noise
// scales to about +/-2 dps. Restore 4.5 once the estimator is cleaned up.
static const double KP_ROLL       = 1.0,  KI_ROLL  = 0.0, KD_ROLL  = 0.0;
static const double KP_PITCH      = 4.5,  KI_PITCH = 0.0, KD_PITCH = 0.0;
static const double KP_YAW        = 2.0,  KI_YAW   = 0.0, KD_YAW   = 0.0;
static const double I_MAX         = 50.0;

// Rate command clamp for bench — deliberately conservative
static const double RATE_MAX_CMD  = 60.0;    // deg/s

// Watchdog thresholds
static const double ATT_STALE_S   = 0.50;    // disarm if ATTITUDE > 500 ms old
static const double SAT_LIMIT_S   = 1.00;    // disarm if command saturated > 1 s

static const double MAX_ANGLE_SP  = 15.0;    // stick range in degrees
static const double PWM_CENTER    = 1500.0;
static const double PWM_RANGE     = 500.0;
static const double PITCH_SP_SIGN = -1.0;

// Sign of the MPU6050's roll axis relative to the FC's (ArduPilot: positive
// roll = right side down). The 2026-09-02 run at KP=3.0 diverged into the end
// stops with corr(roll_cf, ekf_roll) = -0.48: the board reads roll inverted, so
// cmd = KP*(sp - cf) was positive feedback and every correction drove the rig
// further out until the FC's crash detector disarmed at AngErr 87 deg.
// Applied to BOTH the gyro and accel terms, which agree with each other and
// disagree only with the FC. Verify with --monitor before arming.
static const double ROLL_SIGN     = -1.0;

static const uint8_t MY_SYSID     = 254;
static const uint8_t MY_COMPID    = MAV_COMP_ID_ONBOARD_COMPUTER;   // 191

// ArduCopter mode number — verify with your firmware version if in doubt
static const uint32_t MODE_GUIDED_NOGPS = 20;

// SET_ATTITUDE_TARGET type_mask:
//   bit 7 (128) = ignore attitude/quaternion  → we send only rates + throttle
static const uint8_t  TYPE_MASK_RATES_ONLY = 0b10000000;

static const int      CALIB_SAMPLES = 400;

// ---- Scripted setpoint profile (--profile) --------------------------------
// For the baseline-vs-load comparison the two runs must see identical inputs.
// A hand-held stick cannot deliver that: the variation between one thumb
// movement and the next is larger than the timing effect being measured. With
// --profile the stick is ignored and the setpoint follows a fixed square wave,
// so every run is the same length and the same shape, and the only difference
// between them is the load.
//
//   [settle @ 0 deg] then cycles x [ +amp | 0 | -amp | 0 ], each held 'hold' s
//
// The settle window covers spool-up: the FC needs ~1.8 s after arming before
// land_complete clears and it begins acting on rate commands, so stepping any
// earlier would put the first edge into a dead loop.
static bool   g_profile      = false;
static bool   g_monitor      = false;   // --monitor: passive, never arms
static double g_prof_amp     = 10.0;   // deg, clamped to MAX_ANGLE_SP
static double g_prof_hold    = 3.0;    // s per segment
static double g_prof_settle  = 4.0;    // s at zero before the first step
static int    g_prof_cycles  = 3;

// ============================================================================
// GLOBALS
// ============================================================================

enum Axis { AX_ROLL, AX_PITCH, AX_YAW };

static volatile sig_atomic_t g_run     = 1;
static volatile sig_atomic_t g_disarm  = 0;   // set by signal handler
static void on_sigint(int) { g_run = 0; g_disarm = 1; }

static int  g_i2c  = -1;
static int  g_sock = -1;
static volatile bool g_tcp_closed = false;    // set by mav_poll on peer close

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
// MPU6050  (unchanged from rung 1 / UDP version)
// ============================================================================

struct Imu { double ax, ay, az, gx, gy, gz; };

static bool mpu_write(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = { reg, val };
    return write(g_i2c, buf, 2) == 2;
}
static bool mpu_init()
{
    g_i2c = open(I2C_DEV, O_RDWR);
    if (g_i2c < 0) return false;
    if (ioctl(g_i2c, I2C_SLAVE, MPU_ADDR) < 0) return false;
    if (!mpu_write(0x6B, 0x00)) return false;
    usleep(50000);
    if (!mpu_write(0x1B, 0x08)) return false;
    if (!mpu_write(0x1C, 0x00)) return false;
    if (!mpu_write(0x1A, 0x03)) return false;
    usleep(50000);
    return true;
}
static bool mpu_read(Imu* o)
{
    uint8_t reg = 0x3B;
    if (write(g_i2c, &reg, 1) != 1) return false;
    uint8_t d[14];
    if (read(g_i2c, d, 14) != 14) return false;
    auto w = [&](int hi) -> int16_t {
        return (int16_t)(((uint16_t)d[hi] << 8) | d[hi + 1]);
    };
    const double AS = 2.0 / 32768.0, GS = 500.0 / 32768.0;
    o->ax = w(0) * AS;  o->ay = w(2) * AS;  o->az = w(4) * AS;
    o->gx = w(8) * GS;  o->gy = w(10) * GS; o->gz = w(12) * GS;
    return true;
}
static inline double roll_from_accel (const Imu& m) { return atan2(m.ay, m.az) * 180.0 / M_PI; }
static inline double pitch_from_accel(const Imu& m) { return atan2(m.ax, sqrt(m.ay*m.ay + m.az*m.az)) * 180.0 / M_PI; }

// ============================================================================
// MAVLINK
// ============================================================================

struct Telem {
    double roll_sp = 0, pitch_sp = 0, yaw_sp = 0;
    double ekf_roll = 0, ekf_pitch = 0, ekf_yaw = 0;
    double fc_rate_roll = 0, fc_rate_pitch = 0, fc_rate_yaw = 0;
    uint32_t mode = 0;
    bool     armed = false;
    double   t_hb = 0, t_att = 0, t_tgt = 0, t_rc = 0, t_sys = 0;
    uint8_t  sysid = 0, compid = 0;
    bool     have_hb = false;
    long     n_att = 0, n_tgt = 0, n_rc = 0;
};

// Endpoint spec, as passed on the command line:
//   tcp:HOST:PORT   connect to a mavlink-router TcpEndpoint (Mode = Server)
struct EndpointSpec {
    std::string host = "127.0.0.1";
    int         port = TCP_PORT;
    std::string raw;
};

static bool parse_endpoint(const std::string& s, EndpointSpec* e)
{
    e->raw = s;

    size_t c1 = s.find(':');
    if (c1 == std::string::npos) return false;

    std::string scheme = s.substr(0, c1);
    std::string rest   = s.substr(c1 + 1);

    if (scheme != "tcp") return false;

    size_t c2 = rest.find(':');
    if (c2 == std::string::npos) return false;   // tcp always needs HOST:PORT

    e->host = rest.substr(0, c2);
    e->port = atoi(rest.substr(c2 + 1).c_str());

    return e->port > 0 && e->port < 65536 && !e->host.empty();
}

static bool tcp_init(const EndpointSpec& ep)
{
    g_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (g_sock < 0) { perror("socket"); return false; }

    struct sockaddr_in peer;
    memset(&peer, 0, sizeof(peer));
    peer.sin_family = AF_INET;
    peer.sin_port   = htons((uint16_t)ep.port);
    if (inet_pton(AF_INET, ep.host.c_str(), &peer.sin_addr) != 1) {
        fprintf(stderr, "bad endpoint address '%s'\n", ep.host.c_str());
        return false;
    }

    printf("Connecting to mavlink-router TcpEndpoint %s:%d ...\n", ep.host.c_str(), ep.port);
    if (connect(g_sock, (struct sockaddr*)&peer, sizeof(peer)) < 0) {
        perror("connect");
        fprintf(stderr,
            "  Is mavlink-router running with a TcpEndpoint on port %d?\n"
            "  Check with:  ss -ltnp | grep %d\n",
            ep.port, ep.port);
        return false;
    }

    // TCP_NODELAY — MAVLink packets are tiny; we don't want Nagle merging them.
    int one = 1;
    setsockopt(g_sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    // Now switch to non-blocking so mav_poll doesn't stall the control loop.
    int flags = fcntl(g_sock, F_GETFL, 0);
    fcntl(g_sock, F_SETFL, flags | O_NONBLOCK);

    printf("Endpoint: tcp -> %s:%d (connected)\n", ep.host.c_str(), ep.port);
    return true;
}

static void mav_send(const mavlink_message_t* msg)
{
    if (g_sock < 0 || g_tcp_closed) return;
    uint8_t buf[MAVLINK_MAX_PACKET_LEN];
    uint16_t len = mavlink_msg_to_send_buffer(buf, msg);

    // On a non-blocking TCP socket, a full kernel buffer returns EAGAIN. We
    // drop the packet in that case rather than block the loop — MAVLink
    // streams at 50 Hz are self-repairing, and blocking is a worse failure
    // mode. MSG_NOSIGNAL suppresses SIGPIPE if the peer has gone away.
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(g_sock, buf + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) return;   // kernel buf full, drop
            if (errno == EPIPE  || errno == ECONNRESET)  { g_tcp_closed = true; return; }
            return;
        }
        sent += n;
    }
}

static inline double pwm_to_angle(uint16_t pwm)
{
    if (pwm < 800 || pwm > 2200) return 0.0;
    double a = ((double)pwm - PWM_CENTER) / PWM_RANGE * MAX_ANGLE_SP;
    if (a >  MAX_ANGLE_SP) a =  MAX_ANGLE_SP;
    if (a < -MAX_ANGLE_SP) a = -MAX_ANGLE_SP;
    return a;
}

static void mav_poll(Telem* t)
{
    if (g_sock < 0 || g_tcp_closed) return;
    uint8_t buf[2048];
    for (;;) {
        ssize_t n = recv(g_sock, buf, sizeof(buf), 0);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;   // nothing more to read
            g_tcp_closed = true;   // real error
            break;
        }
        if (n == 0) {              // peer closed the connection
            g_tcp_closed = true;
            break;
        }
        mavlink_message_t msg;
        mavlink_status_t  status;
        for (ssize_t i = 0; i < n; i++) {
            if (!mavlink_parse_char(MAVLINK_COMM_0, buf[i], &msg, &status)) continue;
            double now = mono_now();
            switch (msg.msgid) {
            case MAVLINK_MSG_ID_HEARTBEAT: {
                if (msg.sysid == MY_SYSID) break;
                mavlink_heartbeat_t hb;
                mavlink_msg_heartbeat_decode(&msg, &hb);

                // Only the autopilot's heartbeat carries mode + armed state.
                // mavlink-router forwards heartbeats from every endpoint on the
                // mesh — Mission Planner, MAVProxy, other companions — and those
                // all report custom_mode = 0 and base_mode = 0. Accepting them
                // makes mode/armed flicker at the GCS heartbeat rate, which shows
                // up as "mode = 0" at the gate and as a spurious "FC reports
                // DISARMED" abort a second into the run.
                if (hb.autopilot == MAV_AUTOPILOT_INVALID) break;   // GCS/companion
                if (hb.type == MAV_TYPE_GCS)               break;

                if (!t->have_hb) {
                    t->sysid = msg.sysid; t->compid = msg.compid;
                    t->have_hb = true;
                } else if (msg.sysid != t->sysid || msg.compid != t->compid) {
                    break;   // a second autopilot on the mesh — not our vehicle
                }
                t->mode  = hb.custom_mode;
                t->armed = (hb.base_mode & MAV_MODE_FLAG_SAFETY_ARMED) != 0;
                t->t_hb  = now;
                break;
            }
            case MAVLINK_MSG_ID_SYS_STATUS:
                t->t_sys = now; break;
            case MAVLINK_MSG_ID_RC_CHANNELS: {
                mavlink_rc_channels_t rc;
                mavlink_msg_rc_channels_decode(&msg, &rc);
                t->roll_sp  = pwm_to_angle(rc.chan1_raw);
                t->pitch_sp = PITCH_SP_SIGN * pwm_to_angle(rc.chan2_raw);
                t->yaw_sp   = pwm_to_angle(rc.chan4_raw);
                t->t_rc = now; t->n_rc++;
                break;
            }
            case MAVLINK_MSG_ID_ATTITUDE: {
                mavlink_attitude_t a;
                mavlink_msg_attitude_decode(&msg, &a);
                t->ekf_roll  = a.roll  * 180.0 / M_PI;
                t->ekf_pitch = a.pitch * 180.0 / M_PI;
                t->ekf_yaw   = a.yaw   * 180.0 / M_PI;
                t->t_att = now; t->n_att++;
                break;
            }
            case MAVLINK_MSG_ID_ATTITUDE_TARGET: {
                mavlink_attitude_target_t g;
                mavlink_msg_attitude_target_decode(&msg, &g);
                t->fc_rate_roll  = g.body_roll_rate  * 180.0 / M_PI;
                t->fc_rate_pitch = g.body_pitch_rate * 180.0 / M_PI;
                t->fc_rate_yaw   = g.body_yaw_rate   * 180.0 / M_PI;
                t->t_tgt = now; t->n_tgt++;
                break;
            }
            case MAVLINK_MSG_ID_STATUSTEXT: {
                mavlink_statustext_t st;
                mavlink_msg_statustext_decode(&msg, &st);
                char text[51] = {0};
                memcpy(text, st.text, 50);
                fprintf(stderr, "\n[FC sev=%u] %s\n", st.severity, text);
                break;
            }
            }
        }
    }
}

static void send_heartbeat()
{
    mavlink_message_t msg;
    mavlink_msg_heartbeat_pack(MY_SYSID, MY_COMPID, &msg,
        MAV_TYPE_ONBOARD_CONTROLLER, MAV_AUTOPILOT_INVALID,
        0, 0, MAV_STATE_ACTIVE);
    mav_send(&msg);
}

static void send_arm(const Telem& t, bool arm)
{
    mavlink_message_t msg;
    mavlink_msg_command_long_pack(MY_SYSID, MY_COMPID, &msg,
        t.sysid, t.compid,
        MAV_CMD_COMPONENT_ARM_DISARM, 0,
        arm ? 1.0f : 0.0f, 21196.0f,   // 21196 = force flag; needed to arm without RC
        0, 0, 0, 0, 0);
    mav_send(&msg);
}

static void send_setpoint(const Telem& t, double roll_dps, double pitch_dps, double yaw_dps, double thrust)
{
    mavlink_message_t msg;
    float q[4] = { 1.0f, 0.0f, 0.0f, 0.0f };   // identity, ignored by type_mask
    uint32_t tboot = (uint32_t)(mono_now() * 1000.0);
    mavlink_msg_set_attitude_target_pack(MY_SYSID, MY_COMPID, &msg,
        tboot, t.sysid, t.compid,
        TYPE_MASK_RATES_ONLY,
        q,
        (float)(roll_dps  * M_PI / 180.0),
        (float)(pitch_dps * M_PI / 180.0),
        (float)(yaw_dps   * M_PI / 180.0),
        (float)thrust,
        NULL);
    mav_send(&msg);
}

static void request_stream(const Telem& t, uint32_t msgid, double hz)
{
    mavlink_message_t msg;
    mavlink_msg_command_long_pack(MY_SYSID, MY_COMPID, &msg,
        t.sysid, t.compid,
        MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        (float)msgid, (float)(1e6 / hz), 0, 0, 0, 0, 0);
    mav_send(&msg);
}

// ============================================================================
// PID
// ============================================================================

struct Pid {
    double kp, ki, kd, integral = 0.0, prev_meas = 0.0;
    bool first = true;
    double p = 0, i = 0, d = 0;
    double step(double sp, double meas, double dt)
    {
        double err = sp - meas;
        p = kp * err;
        integral += err * dt;
        double icap = (ki > 0.0) ? I_MAX / ki : I_MAX;
        if (integral >  icap) integral =  icap;
        if (integral < -icap) integral = -icap;
        i = ki * integral;
        if (first) { prev_meas = meas; first = false; }
        d = -kd * (meas - prev_meas) / (dt > 0 ? dt : DT_NOMINAL);
        prev_meas = meas;
        double out = p + i + d;
        if (out >  RATE_MAX_CMD) out =  RATE_MAX_CMD;
        if (out < -RATE_MAX_CMD) out = -RATE_MAX_CMD;
        return out;
    }
};

// Scripted setpoint in degrees for elapsed run time t. Sets *done once the
// profile has played out, which ends the run at a deterministic length so the
// baseline and loaded logs cover exactly the same manoeuvre.
static double profile_setpoint(double t, bool* done)
{
    *done = false;
    if (t < g_prof_settle) return 0.0;

    const double period = 4.0 * g_prof_hold;
    const double u      = t - g_prof_settle;
    if (u >= g_prof_cycles * period) { *done = true; return 0.0; }

    const double ph = fmod(u, period);
    if (ph <     g_prof_hold) return  g_prof_amp;
    if (ph < 2 * g_prof_hold) return  0.0;
    if (ph < 3 * g_prof_hold) return -g_prof_amp;
    return 0.0;
}

// ============================================================================
// SAFE SHUTDOWN
// ============================================================================

static void safe_disarm(const Telem& tel, const char* reason)
{
    fprintf(stderr, "\n*** DISARM: %s ***\n", reason);
    // Zero the rates and send disarm several times; the FC may drop packets.
    // If the TCP peer has already closed, mav_send will short-circuit and the
    // FC will still disarm itself via GUID_TIMEOUT after 3 s.
    for (int i = 0; i < 10; i++) {
        send_setpoint(tel, 0, 0, 0, 0.0);
        send_arm(tel, false);
        usleep(50000);
        mav_poll((Telem*)&tel);   // drain replies
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <label> <roll|pitch|yaw> <throttle 0..0.6> [endpoint]\n"
                        "  endpoint defaults to tcp:127.0.0.1:%d and accepts:\n"
                        "    tcp:HOST:PORT   connect to a mavlink-router TcpEndpoint\n",
                argv[0], TCP_PORT);
        return 2;
    }
    std::string label   = argv[1];
    std::string axisstr = argv[2];
    double      thr     = atof(argv[3]);
    Axis axis;
    if      (axisstr == "roll")  axis = AX_ROLL;
    else if (axisstr == "pitch") axis = AX_PITCH;
    else if (axisstr == "yaw")   axis = AX_YAW;
    else { fprintf(stderr, "axis must be roll|pitch|yaw\n"); return 2; }
    if (thr < 0.0 || thr > 0.60) {
        fprintf(stderr, "throttle out of allowed bench range 0.00..0.60\n"); return 2;
    }

    // Optional trailing args: --profile and its parameters may appear in any
    // order alongside the endpoint.
    std::string epstr = DEFAULT_ENDPOINT;
    for (int i = 4; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--profile")               g_profile = true;
        else if (a == "--monitor")               g_monitor = true;
        else if (a.rfind("--amp=",    0) == 0)   g_prof_amp    = atof(a.c_str() + 6);
        else if (a.rfind("--hold=",   0) == 0)   g_prof_hold   = atof(a.c_str() + 7);
        else if (a.rfind("--cycles=", 0) == 0)   g_prof_cycles = atoi(a.c_str() + 9);
        else if (a.rfind("--settle=", 0) == 0)   g_prof_settle = atof(a.c_str() + 9);
        else if (a.rfind("--", 0) == 0) {
            fprintf(stderr, "unknown option %s\n", a.c_str()); return 2;
        }
        else epstr = a;
    }
    if (g_prof_amp > MAX_ANGLE_SP) {
        fprintf(stderr, "--amp clamped from %.1f to stick range %.1f deg\n",
                g_prof_amp, MAX_ANGLE_SP);
        g_prof_amp = MAX_ANGLE_SP;
    }
    if (g_prof_amp < 0.0 || g_prof_hold <= 0.0 || g_prof_cycles < 1 || g_prof_settle < 0.0) {
        fprintf(stderr, "invalid profile parameters\n"); return 2;
    }

    EndpointSpec ep;
    if (!parse_endpoint(epstr, &ep)) {
        fprintf(stderr, "bad endpoint '%s' (expected tcp:HOST:PORT)\n", epstr.c_str());
        return 2;
    }

    signal(SIGINT,  on_sigint);
    signal(SIGTERM, on_sigint);
    signal(SIGPIPE, SIG_IGN);     // never let a dead peer kill us

    char stamp[32]; time_t wall = time(NULL);
    strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", localtime(&wall));
    std::string base     = std::string("/dev/shm/") + label + "_" + stamp;
    std::string csv_path = base + ".csv";
    std::string meta_path = base + ".meta.txt";

    struct sched_param sp; sp.sched_priority = 80;
    bool have_fifo = (sched_setscheduler(0, SCHED_FIFO, &sp) == 0);
    printf("SCHED_FIFO: %s\n", have_fifo ? "yes" : "no (run with sudo)");

    if (!mpu_init())    { fprintf(stderr, "MPU init failed\n"); return 1; }
    if (!tcp_init(ep))  { fprintf(stderr, "TCP init failed\n"); return 1; }

    // Wait for the FC's first heartbeat. Unlike the UDP Server-endpoint case
    // we don't have to prime the router to remember us — connect() already
    // did that. But we still need the FC's sysid/compid before we can send
    // COMMAND_LONG to it. We also send our own heartbeat so any GCS on the
    // same router (e.g. Mission Planner over the SSH tunnel) sees us.
    printf("Waiting for heartbeat on %s ...\n", ep.raw.c_str());
    Telem tel;
    double hb_wait_next = 0.0;
    while (g_run && !tel.have_hb && !g_tcp_closed) {
        double now = mono_now();
        if (now >= hb_wait_next) { send_heartbeat(); hb_wait_next = now + 0.2; }
        mav_poll(&tel);
        usleep(20000);
    }
    if (!g_run) return 0;
    if (g_tcp_closed) { fprintf(stderr, "TCP peer closed before first heartbeat\n"); return 1; }
    printf("Connected — vehicle sysid %u, compid %u\n", tel.sysid, tel.compid);

    // ---- Ensure we can see attitude + target at 50 Hz ----
    for (int i = 0; i < 3; i++) {
        request_stream(tel, MAVLINK_MSG_ID_ATTITUDE,        50.0);
        request_stream(tel, MAVLINK_MSG_ID_ATTITUDE_TARGET, 50.0);
        request_stream(tel, MAVLINK_MSG_ID_RC_CHANNELS,     50.0);
        usleep(100000);
        mav_poll(&tel);
    }

    // ---- MONITOR MODE ----
    // Passive. Never arms, never sends a setpoint, ignores the mode gate. Tilt
    // the rig by hand and confirm roll_cf moves the SAME direction as ekf_roll.
    // This check is impossible on a props-off run because the rig never moves,
    // which is how an inverted axis reached a closed loop on 2026-09-02.
    if (g_monitor) {
        printf("\nMONITOR — no arming, no setpoints sent.\n");
        printf("Hold the rig still, calibrating...\n");
        sleep(2);
        double gx_o = 0, r_o = 0;
        for (int i = 0; i < CALIB_SAMPLES; i++) {
            Imu m;
            if (mpu_read(&m)) { gx_o += m.gx; r_o += roll_from_accel(m); }
            usleep(5000);
        }
        gx_o /= CALIB_SAMPLES; r_o /= CALIB_SAMPLES;
        printf("Offsets: gx %.2f  roll %.2f\n\n", gx_o, r_o);
        printf("Tilt the rig by hand. cf and ekf must move the SAME way and\n"
               "roughly the same amount. ROLL_SIGN is currently %+.0f.\n\n", ROLL_SIGN);
        printf("%12s %12s %12s %12s\n", "accel_roll", "gyro_dps", "roll_cf", "ekf_roll");
        double cfr = 0, prev = mono_now();
        int k = 0;
        while (g_run && !g_tcp_closed) {
            Imu m;
            if (!mpu_read(&m)) { fprintf(stderr, "\nIMU read failed\n"); break; }
            double now = mono_now(), d = now - prev; prev = now;
            if (d <= 0.0 || d > 0.5) d = DT_NOMINAL;
            double gx   = ROLL_SIGN * (m.gx - gx_o);
            double racc = ROLL_SIGN * (roll_from_accel(m) - r_o);
            cfr = ALPHA * (cfr + gx * d) + (1 - ALPHA) * racc;
            mav_poll(&tel);
            if (++k % 10 == 0) {
                printf("\r%12.2f %12.2f %12.2f %12.2f   ", racc, gx, cfr, tel.ekf_roll);
                fflush(stdout);
            }
            usleep(20000);
        }
        printf("\nMonitor ended — nothing was armed.\n");
        return 0;
    }

    // ---- MODE GATE ----
    printf("\nFC reports mode = %u  (need %u = GUIDED_NOGPS)\n", tel.mode, MODE_GUIDED_NOGPS);
    if (tel.mode != MODE_GUIDED_NOGPS) {
        fprintf(stderr, "REFUSING to arm: set flight mode to GUIDED_NOGPS in Mission Planner first.\n");
        return 3;
    }

    // ---- USER GATE ----
    printf("\nStage summary:\n");
    printf("  axis under command : %s (others held at 0 dps)\n", axisstr.c_str());
    printf("  throttle           : %.2f\n", thr);
    printf("  rate clamp         : +/- %.1f dps\n", RATE_MAX_CMD);
    if (g_profile) {
        double dur = g_prof_settle + g_prof_cycles * 4.0 * g_prof_hold;
        printf("  setpoint           : SCRIPTED (stick ignored)\n");
        printf("  profile            : %.1f s settle, %d x [+%.1f | 0 | -%.1f | 0] @ %.1f s\n",
               g_prof_settle, g_prof_cycles, g_prof_amp, g_prof_amp, g_prof_hold);
        printf("  run duration       : %.1f s, then auto-disarm\n", dur);
    } else {
        printf("  setpoint           : RC stick (not repeatable between runs)\n");
    }
    printf("\nBefore continuing:\n");
    printf("  [ ] props correctly installed (or off for Stage A)?\n");
    printf("  [ ] rig mechanically locks non-commanded axes?\n");
    printf("  [ ] mechanical end stops in place at +/- 30 deg?\n");
    printf("  [ ] Mission Planner disarm button visible on laptop?\n");
    printf("  [ ] helper standing by, hand on battery lead?\n");
    printf("\nType READY and press ENTER to proceed, anything else to abort: ");
    fflush(stdout);
    char answer[32] = {0};
    if (!fgets(answer, sizeof(answer), stdin) || strncmp(answer, "READY", 5) != 0) {
        fprintf(stderr, "Aborted.\n"); return 4;
    }

    // ---- Calibration ----
    printf("Hold rig still — calibrating (%d samples)...\n", CALIB_SAMPLES); sleep(2);
    double gx_off=0, gy_off=0, gz_off=0, roll_off=0, pitch_off=0;
    for (int i = 0; i < CALIB_SAMPLES; i++) {
        Imu m; if (!mpu_read(&m)) { fprintf(stderr, "IMU read failed in calib\n"); return 1; }
        gx_off += m.gx; gy_off += m.gy; gz_off += m.gz;
        roll_off += roll_from_accel(m); pitch_off += pitch_from_accel(m);
        usleep(5000);
    }
    gx_off /= CALIB_SAMPLES; gy_off /= CALIB_SAMPLES; gz_off /= CALIB_SAMPLES;
    roll_off /= CALIB_SAMPLES; pitch_off /= CALIB_SAMPLES;
    printf("Offsets: roll %.2f pitch %.2f  gx %.2f gy %.2f gz %.2f\n",
           roll_off, pitch_off, gx_off, gy_off, gz_off);

    // ---- Metadata ----
    {
        FILE* mf = fopen(meta_path.c_str(), "w");
        if (mf) {
            char wt[64]; time_t w=time(NULL);
            strftime(wt, sizeof(wt), "%Y-%m-%d %H:%M:%S %Z", localtime(&w));
            fprintf(mf, "run_label     : %s\nstarted       : %s\n", label.c_str(), wt);
            fprintf(mf, "mode          : ACTUATION (rung 2, TCP)\naxis          : %s\n", axisstr.c_str());
            fprintf(mf, "throttle      : %.2f\nrate_clamp    : %.1f dps\n", thr, RATE_MAX_CMD);
            fprintf(mf, "endpoint      : %s\n", ep.raw.c_str());
            fprintf(mf, "alpha         : %.4f\n", ALPHA);
            fprintf(mf, "gains         : roll Kp=%.2f  pitch Kp=%.2f  yaw Kp=%.2f\n",
                    KP_ROLL, KP_PITCH, KP_YAW);
            fprintf(mf, "sched_fifo    : %s\n", have_fifo ? "yes" : "no");
            fprintf(mf, "calib_offsets : roll %.2f pitch %.2f gx %.2f gy %.2f gz %.2f\n",
                    roll_off, pitch_off, gx_off, gy_off, gz_off);
            fclose(mf);
        }
    }

    // ---- Log ----
    FILE* log = fopen(csv_path.c_str(), "w");
    if (!log) { perror("fopen"); return 1; }
    static char logbuf[1<<20]; setvbuf(log, logbuf, _IOFBF, sizeof(logbuf));
    fprintf(log,
        "run_label,seq,t_s,dt_s,compute_us,axis,armed,mode,"
        "roll_cf,pitch_cf,yaw_cf,"
        "roll_sp,pitch_sp,yaw_sp,"
        "roll_cmd,pitch_cmd,yaw_cmd,thrust,"
        "ekf_roll,ekf_pitch,ekf_yaw,"
        "fc_rate_roll,fc_rate_pitch,fc_rate_yaw,"
        "age_att,age_tgt,age_hb,saturated\n");

    // ---- Arm ----
    printf("\nArming in 2 s...\n"); sleep(2);
    for (int i = 0; i < 5 && !tel.armed; i++) {
        send_arm(tel, true); usleep(200000); mav_poll(&tel);
    }
    if (!tel.armed) {
        fprintf(stderr, "Arming failed — check ARMING_CHECK and mode.\n");
        return 5;
    }
    printf("Armed. Ctrl-C to stop.\n\n");

    double roll_cf = 0, pitch_cf = 0, yaw_cf = 0;
    Pid pid_r{KP_ROLL, KI_ROLL, KD_ROLL};
    Pid pid_p{KP_PITCH, KI_PITCH, KD_PITCH};
    Pid pid_y{KP_YAW,   KI_YAW,   KD_YAW};

    struct timespec next; clock_gettime(CLOCK_MONOTONIC, &next);
    double t0 = mono_now(), prev_wake = t0;
    double sat_since = -1.0, hb_next = t0 + 1.0;
    long seq = 0;
    const char* abort_reason = nullptr;
    bool profile_complete = false;

    while (g_run) {
        ts_add_ns(&next, PERIOD_NS);
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
        double wake = mono_now();
        double dt = wake - prev_wake; prev_wake = wake;
        if (dt <= 0.0) dt = DT_NOMINAL;
        if (dt >  0.5) dt = 0.5;
        double c0 = mono_now();

        // Sensor
        Imu m;
        if (!mpu_read(&m)) { abort_reason = "IMU read failed"; break; }
        m.gx -= gx_off; m.gy -= gy_off; m.gz -= gz_off;
        double racc = roll_from_accel(m)  - roll_off;
        double pacc = pitch_from_accel(m) - pitch_off;
        roll_cf  = ALPHA*(roll_cf  + ROLL_SIGN*m.gx*dt) + (1-ALPHA)*(ROLL_SIGN*racc);
        pitch_cf = ALPHA*(pitch_cf - m.gy*dt) + (1-ALPHA)*pacc;
        yaw_cf  += m.gz * dt;   // yaw has no accel reference — drift acceptable on bench

        // Telemetry
        mav_poll(&tel);

        // Scripted setpoint overrides the stick, so baseline and loaded runs
        // see an identical input and identical duration.
        if (g_profile) {
            bool prof_done = false;
            double psp = profile_setpoint(wake - t0, &prof_done);
            if (prof_done) { profile_complete = true; break; }
            tel.roll_sp = tel.pitch_sp = tel.yaw_sp = 0.0;
            if      (axis == AX_ROLL)  tel.roll_sp  = psp;
            else if (axis == AX_PITCH) tel.pitch_sp = psp;
            else                       tel.yaw_sp   = psp;
        }

        // Watchdogs
        if (g_tcp_closed)                                       { abort_reason = "TCP peer closed"; break; }
        if (!tel.armed)                                         { abort_reason = "FC reports DISARMED"; break; }
        if (tel.mode != MODE_GUIDED_NOGPS)                      { abort_reason = "mode left GUIDED_NOGPS"; break; }
        if (tel.t_att > 0 && (wake - tel.t_att) > ATT_STALE_S)  { abort_reason = "ATTITUDE stream stalled"; break; }

        // Compute PID — only the chosen axis carries a real command
        double cmd_r = pid_r.step(tel.roll_sp,  roll_cf,  dt);
        double cmd_p = pid_p.step(tel.pitch_sp, pitch_cf, dt);
        double cmd_y = pid_y.step(tel.yaw_sp,   yaw_cf,   dt);
        double send_r = 0, send_p = 0, send_y = 0;
        if (axis == AX_ROLL)  send_r = cmd_r;
        if (axis == AX_PITCH) send_p = cmd_p;
        if (axis == AX_YAW)   send_y = cmd_y;

        double chosen = (axis==AX_ROLL) ? cmd_r : (axis==AX_PITCH ? cmd_p : cmd_y);
        int saturated = (fabs(chosen) >= RATE_MAX_CMD - 0.5) ? 1 : 0;
        if (saturated) {
            if (sat_since < 0) sat_since = wake;
            else if (wake - sat_since > SAT_LIMIT_S) { abort_reason = "command saturated too long"; break; }
        } else sat_since = -1.0;

        // Actuate
        send_setpoint(tel, send_r, send_p, send_y, thr);

        // 1 Hz heartbeat
        if (wake >= hb_next) { send_heartbeat(); hb_next = wake + 1.0; }

        double compute_us = (mono_now() - c0) * 1e6;

        fprintf(log,
            "%s,%ld,%.6f,%.6f,%.1f,%s,%d,%u,"
            "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
            "%.3f,%.3f,%.3f,%.3f,"
            "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
            "%.3f,%.3f,%.3f,%d\n",
            label.c_str(), seq, wake-t0, dt, compute_us,
            axisstr.c_str(), tel.armed?1:0, tel.mode,
            roll_cf, pitch_cf, yaw_cf,
            tel.roll_sp, tel.pitch_sp, tel.yaw_sp,
            send_r, send_p, send_y, thr,
            tel.ekf_roll, tel.ekf_pitch, tel.ekf_yaw,
            tel.fc_rate_roll, tel.fc_rate_pitch, tel.fc_rate_yaw,
            tel.t_att>0 ? wake-tel.t_att : -1.0,
            tel.t_tgt>0 ? wake-tel.t_tgt : -1.0,
            tel.t_hb >0 ? wake-tel.t_hb  : -1.0,
            saturated);

        if (seq % 10 == 0) {
            printf("\rt %5.1f  sp %+6.1f  cf %+6.1f  ekf %+6.1f  cmd %+6.1f dps  "
                   "fc %+6.1f  age_att %.2f  mode=%u %s   ",
                   wake-t0,
                   (axis==AX_ROLL?tel.roll_sp:(axis==AX_PITCH?tel.pitch_sp:tel.yaw_sp)),
                   (axis==AX_ROLL?roll_cf:(axis==AX_PITCH?pitch_cf:yaw_cf)),
                   (axis==AX_ROLL?tel.ekf_roll:(axis==AX_PITCH?tel.ekf_pitch:tel.ekf_yaw)),
                   chosen,
                   (axis==AX_ROLL?tel.fc_rate_roll:(axis==AX_PITCH?tel.fc_rate_pitch:tel.fc_rate_yaw)),
                   tel.t_att>0 ? wake-tel.t_att : -1.0,
                   tel.mode,
                   saturated ? "SAT" : "   ");
            fflush(stdout);
        }
        seq++;
    }

    // ---- Shutdown path — always disarm ----
    safe_disarm(tel, abort_reason ? abort_reason
                                  : profile_complete ? "profile complete"
                                                     : "user Ctrl-C");
    fflush(log); fclose(log);
    if (g_i2c>=0) close(g_i2c);
    if (g_sock>=0) close(g_sock);
    double elapsed = mono_now() - t0;
    printf("\n\nStopped after %.1f s, %ld cycles.\n", elapsed, seq);
    printf("Log: %s\n", csv_path.c_str());
    return abort_reason ? 6 : 0;
}
