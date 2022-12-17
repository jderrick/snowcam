import time
import vail

if __name__ == '__main__':
    # Runs every 5 minutes, avoiding drift
    start_time = time.time()
    while True:
        vail.run_vail()
        time.sleep(300 - ((time.time() - start_time) % 300))
