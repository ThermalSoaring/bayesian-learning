#
# Demo getting data from CRRCSim
#
# Based on:
#    https://github.com/dronekit/dronekit-sitl/tree/master/dronekit_sitl/pysim
#
# Run crrcsim, set Options->Controls->Input Method->MNAV
# so that it'll open localhost:9002 that we can connect to.
#


from crrcsim import CRRCSim

sim = CRRCSim()

while True:
    #sim.recv_fdm()

    print "Accel_Body:", sim.accel_body, "\n", \
        "Gyro:", sim.gyro, "\n", \
        "Velocity:", sim.velocity, "\n", \
        "PosX:", sim.position.x, "\n", \
        "PosY:", sim.position.y, "\n", \
        "PosZ:", sim.position.z, "\n", \
        "DCM:", sim.dcm, "\n", \
        "Time:", sim.time_now, "\n"

    #roll_rate = 0.0
    #pitch_rate = 0.0
    #throttle = 100.0
    #yaw_rate = 0.0
    #sim.update([roll_rate, pitch_rate, throttle, yaw_rate])

    #aileron = 0.0
    #elevator = 0.0
    #throttle = 100.0
    #junk = 0.0
    #sim.update([aileron, elevator, throttle, junk])

    sim.update([0,0,0,0])
