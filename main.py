import argparse
from fai_video_streamer import FaiNumberPlateFetcher
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Provide the device address (int, str).')
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        default=0,
        help='Choose the device',
    )

    args = parser.parse_args()

    capture_device = int(args.device) if str(
        args.device).isdigit() else str(args.device)
    while True:
        plate_fetcher = None
        try:
            plate_fetcher = FaiNumberPlateFetcher(
                capture_device=capture_device)
            plate_fetcher.start()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)
            if plate_fetcher:
                plate_fetcher.stop()
            continue
        except KeyboardInterrupt:
            break
