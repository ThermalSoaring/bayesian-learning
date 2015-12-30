import time
from FlightGear import FlightGear

class System:
    def __init__(self, server='localhost', port=5500):
        self.fg = FlightGear(server, port)

        # Set initial values
        self.update()

    def update(self):
        # Orientation
        #self.heading = self.fg['/orientation/heading-deg']
        #self.roll = self.fg['/orientation/roll-deg']
        #self.pitch = self.fg['/orientation/pitch-deg']
        #self.yaw = self.fg['/orientation/yaw-deg']
        #self.roll_rate = self.fg['/orientation/roll-rate-degps']
        #self.pitch_rate = self.fg['/orientation/pitch-rate-degps']
        #self.yaw_rate = self.fg['/orientation/yaw-rate-degps']

        # Position
        self.longitude = self.fg['/position/longitude-deg']
        self.latitude = self.fg['/position/latitude-deg']
        self.altitude = self.fg['/position/altitude-ft']
        #self.altitude_agl = self.fg['/position/altitude-agl-ft']
        #self.ground_elev = self.fg['/position/ground-elev-ft']

        # Velocity
        #self.uBody = self.fg['/velocities/uBody-fps']
        #self.vBody = self.fg['/velocities/vBody-fps']
        #self.wBody = self.fg['/velocities/wBody-fps']
        #self.north_speed = self.fg['/velocities/speed-north-fps']
        #self.east_speed = self.fg['/velocities/speed-east-fps']
        #self.down_speed = self.fg['/velocities/speed-down-fps']
        #self.north_speed_relgrnd = self.fg['/velocities/north-relground-fps']
        #self.east_speed_relgrnd = self.fg['/velocities/east-relground-fps']
        #self.down_speed_relgrnd = self.fg['/velocities/down-relground-fps']
        self.airspeed = self.fg['/velocities/airspeed-kt']
        #self.groundspeed = self.fg['/velocities/groundspeed-kt']
        self.vert_speed = self.fg['/velocities/vertical-speed-fps']
        #self.glideslope = self.fg['/velocities/glideslope']

        # Acceleration
        #self.north_accel = self.fg['/accelerations/ned/north-accel-fps_sec']
        #self.east_accel = self.fg['/accelerations/ned/east-accel-fps_sec']
        #self.down_accel = self.fg['/accelerations/ned/down-accel-fps_sec']

        # Environment
        #self.pressure_sealevel = self.fg['/environment/pressure-sea-level-inhg']
        #self.pressure = self.fg['/environment/pressure-inhg']
        self.temperature = self.fg['/environment/temperature-degf']
        #self.wind_speed = self.fg['/environment/wind-speed-kt']
        #self.wind_from_heading = self.fg['/environment/wind-from-heading-deg']
        #self.wind_from_north = self.fg['/environment/wind-from-north-fps']
        #self.wind_from_east = self.fg['/environment/wind-from-east-fps']
        #self.wind_from_down = self.fg['/environment/wind-from-down-fps']

        # Pressure sensors
        self.total_pressure = self.fg['/systems/pitot/total-pressure-inhg']
        self.measured_total_pressure = self.fg['/systems/pitot/measured-total-pressure-inhg']
        self.static_pressure = self.fg['/systems/static/pressure-inhg'] # Same as pressure?

    def __repr__(self):
        return "Position\n" + \
        "\tLongitude: " + str(self.longitude) + " deg\n" + \
        "\tLatitude: " + str(self.latitude) + " deg\n" + \
        "\tAltitude: " + str(self.altitude) + " ft\n" + \
        "Velocity\n" + \
        "\tAirSpeed: " + str(self.airspeed) + " kt\n" + \
        "\tVertSpeed: " + str(self.vert_speed) + " fps\n" + \
        "Environment\n" + \
        "\tTemp: " + str(self.temperature) + " degf\n" + \
        "Pressure sensors\n" + \
        "\tTotal: " + str(self.total_pressure) + " inhg\n" + \
        "\tMeasuredTotal: " + str(self.measured_total_pressure) + " inhg\n" + \
        "\tStatic: " + str(self.static_pressure) + " inhg\n"

        """
        return
        "Orientation\n" + \
        "\tHeading: " + str(self.heading) + " deg\n" + \
        "\tRoll: " + str(self.roll) + " deg\n" + \
        "\tPitch: " + str(self.pitch) + " deg\n" + \
        "\tYaw: " + str(self.yaw) + " deg\n" + \
        "\tRollRate: " + str(self.roll_rate) + " deg/s\n" + \
        "\tPitchRate: " + str(self.pitch_rate) + " deg/s\n" + \
        "\tYawRate: " + str(self.yaw_rate) + " deg/s\n" + \
        "Position\n" + \
        "\tLongitude: " + str(self.longitude) + " deg\n" + \
        "\tLatitude: " + str(self.latitude) + " deg\n" + \
        "\tAltitude: " + str(self.altitude) + " ft\n" + \
        "\tAltitudeAGL: " + str(self.altitude_agl) + " ft\n" + \
        "\tGroundElev: " + str(self.ground_elev) + " ft\n" + \
        "Velocity\n" + \
        "\tuBody: " + str(self.uBody) + " fps\n" + \
        "\tvBody: " + str(self.vBody) + " fps\n" + \
        "\twBody: " + str(self.wBody) + " fps\n" + \
        "\tNorthSpeed: " + str(self.north_speed) + " fps\n" + \
        "\tEastSpeed: " + str(self.east_speed) + " fps\n" + \
        "\tDownSpeed: " + str(self.down_speed) + " fps\n" + \
        "\tNorthSpeedRG: " + str(self.north_speed_relgrnd) + " fps\n" + \
        "\tEastSpeedRG: " + str(self.east_speed_relgrnd) + " fps\n" + \
        "\tDownSpeedRG: " + str(self.down_speed_relgrnd) + " fps\n" + \
        "\tAirSpeed: " + str(self.airspeed) + " kt\n" + \
        "\tGroundSpeed: " + str(self.groundspeed) + " kt\n" + \
        "\tVertSpeed: " + str(self.vert_speed) + " fps\n" + \
        "\tGlideSlow: " + str(self.glideslope) + " \n" + \
        "Acceleration\n" + \
        "\tNorth: " + str(self.north_accel) + " ft/s^2\n" + \
        "\tEast: " + str(self.east_accel) + " ft/s^2\n" + \
        "\tDown: " + str(self.down_accel) + " ft/s^2\n" + \
        "Environment\n" + \
        "\tPressureSL: " + str(self.pressure_sealevel) + " inhg\n" + \
        "\tPressure: " + str(self.pressure) + " inhg\n" + \
        "\tTemp: " + str(self.temperature) + " degf\n" + \
        "\tWindSpeed: " + str(self.wind_speed) + " kt\n" + \
        "\tWindFromHeading: " + str(self.wind_from_heading) + " deg\n" + \
        "\tWindFromN: " + str(self.wind_from_north) + " fps\n" + \
        "\tWindFromE: " + str(self.wind_from_east) + " fps\n" + \
        "\tWindFromD: " + str(self.wind_from_down) + " fps\n" + \
        "Pressure sensors\n" + \
        "\tTotal: " + str(self.total_pressure) + " inhg\n" + \
        "\tMeasuredTotal: " + str(self.measured_total_pressure) + " inhg\n" + \
        "\tStatic: " + str(self.static_pressure) + " inhg\n"
        """

# parking brake on
#fg['/controls/gear/brake-parking'] = 1

s = System('localhost',5500)
#fg = FlightGear('localhost', 5500)

# Get current heading
while True:
    print s
    s.update()
