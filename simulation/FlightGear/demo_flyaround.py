from FlightGear import FlightGear
import time

def main():
    fg = FlightGear('localhost', 5500)

    # Wait five seconds for simulator to settle down
    while 1:
        if fg['/sim/time/elapsed-sec'] > 5:
            break
        time.sleep(1.0)
        print fg['/sim/time/elapsed-sec']


    # parking brake on
    fg['/controls/parking-brake'] = 1

    heading = fg['/orientation/heading-deg']

    # Switch to external view for for 'walk around'.
    fg.view_next()

    interval = 0.05
    for i in xrange(0, 180, 20):
        fg['/sim/current-view/goal-heading-offset-deg'] = 180.0 - i
        time.sleep(interval)
        print "Sending", 180-i

    time.sleep(2.0)

    # Switch back to cockpit view
    fg.view_prev()

    time.sleep(2.0)

    # Flaps to take off position
    fg['/controls/flaps'] = 0.34
    #fg.wait_for_prop_eq('/surface-positions/flap-pos-norm', 0.34)

    fg.quit()

if __name__ == '__main__':
    main()
