import cv2
import numpy as np
from time import sleep, time
import threading
import signal
import sys
from stereo_pair import StereoCamera
from disparity_computer import DisparityComputer
from enhanced_depth_computer import EnhancedDepthComputer
from mavlink_communication import MAVLinkCommunication
from obstacle_mapper import ObstacleMapper


def signal_handler(sig, frame):
    """Obsługa Ctrl+C"""
    print('\nZatrzymywanie systemu...')
    sys.exit(0)


def main():
    print("Inicjalizacja systemu Stereo Vision Obstacle Avoidance...")

    # Parametry kamery (dostosuj do Twojego setupu)
    focal_length_px = 2714.29
    baseline_m = 0.06

    try:
        # Inicjalizacja komponentów
        stereo_camera = StereoCamera()
        disparity_computer = DisparityComputer()
        depth_computer = EnhancedDepthComputer(focal_length_px, baseline_m)

        # Inicjalizacja MAVLink (dostosuj connection_string)
        mavlink_comm = MAVLinkCommunication(
            connection_string='/dev/ttyACM0',  # lub '/dev/serial0' dla Pi
            baud=57600
        )

        obstacle_mapper = ObstacleMapper(mavlink_comm)

        print("Uruchamianie kamer...")
        stereo_camera.start()
        sleep(2)  # Czas na rozgrzanie kamer

        # Obsługa Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)

        print("System gotowy! Naciśnij 'q' aby zakończyć.")

        frame_count = 0
        start_time = time()

        while True:
            # Pobierz klatki z kamer
            left_frame, right_frame = stereo_camera.get_frames()

            if left_frame is None or right_frame is None:
                continue

            # Oblicz mapę dysparycji
            disparity = disparity_computer.compute(left_frame, right_frame)

            # Oblicz mapę głębi
            depth_map = depth_computer.compute_depth(disparity)

            # Zastosuj filtry
            depth_filtered = depth_computer.apply_filters(depth_map)

            # Wyślij dane o przeszkodach do ArduPilot
            obstacle_mapper.send_obstacle_data(depth_filtered)

            # Wyświetl wyniki (opcjonalne - można wyłączyć dla lepszej wydajności)
            if True:  # Ustaw False aby wyłączyć wyświetlanie
                disp_vis = disparity_computer.normalize(disparity)
                depth_vis = depth_computer.normalize_for_display(depth_filtered)

                # Kombinuj obrazy do wyświetlenia
                combined = np.hstack((
                    cv2.resize(left_frame, (320, 240)),
                    cv2.resize(disp_vis, (320, 240)),
                    cv2.resize(cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET), (320, 240))
                ))

                cv2.imshow("Stereo Vision: Left | Disparity | Depth", combined)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Statystyki FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time() - start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}, Klatka: {frame_count}")

    except KeyboardInterrupt:
        print("Zatrzymano przez użytkownika")
    except Exception as e:
        print(f"Błąd: {e}")
    finally:
        print("Czyszczenie zasobów...")
        try:
            stereo_camera.stop()
            cv2.destroyAllWindows()
        except:
            pass


if __name__ == "__main__":
    main()
