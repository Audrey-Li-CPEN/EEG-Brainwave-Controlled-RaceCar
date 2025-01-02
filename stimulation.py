

# Idea for this stimulation script for Motor Imagery task data recording on user's computer
# This script will generate a stimulation pattern on the screen and record the data in a separate CSV file
# The CSV file will be saved in the same directory as the script
# The CSV file will contain the stimulation pattern, the timestamp, and the user's response

#Steps(Sequence is important to minimize the time discrepency between the stimulation and the recording):
# 1. Set up OpenViBE acquisition server with EPOC+
# 2. Open OpenViBE scenario
# 3. Start the OpenViBE recording (this creates the CSV file)
# 4. Immediately run the stimulation script (python stimulation.py)
# 5. The script will find the CSV file and start writing labels
# 6. When finished, stop the script (ESC key)
# 7. Stop the OpenViBE recording
# We will combine the raw data and the tags csv after recordings

import pygame
import random
import time
from pygame.locals import *
import csv
import os

class StimulationDisplay:
    def __init__(self):
        pygame.init()
        # Set up the display window (full screen recommended for experiments)
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Motor Imagery Stimulation')

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)

        # Stimulation parameters
        self.stim_duration = 4  # Duration of each stimulation in seconds
        self.rest_duration = 2  # Duration of rest period in seconds
        self.patterns = ['rest', 'left', 'right', 'stop']

        # CSV handling
        self.openvibe_output_dir = r"D:\\openvibe-3.6.0-64bit\\share\\openvibe\\scenarios\\bci-examples"
        self.csv_path = os.path.join(self.openvibe_output_dir, "test.csv")
        self.tag_csv_path = os.path.join(self.openvibe_output_dir, "tags.csv")
        self.init_tag_csv()

    def init_tag_csv(self):
        with open(self.tag_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time:128Hz', 'tag'])  # Initialize the header

    def read_current_time_from_csv(self):
        try:
            with open(self.csv_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)  # Read the header
                time_index = header.index('Time:128Hz')  # Locate the time column

                # Get the last recorded time in the CSV
                for row in reader:
                    last_time = row[time_index]

                return last_time
        except Exception as e:
            print(f"Error reading test.csv: {e}")
            return None

    def write_stimulation_tag(self, timestamp, tag):
        try:
            current_time = self.read_current_time_from_csv()
            if current_time is None:
                print("Failed to read current time from test.csv")
                return

            with open(self.tag_csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_time, tag])
        except Exception as e:
            print(f"Error writing to tags.csv: {e}")

    def draw_arrow(self, direction):
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        arrow_size = 100

        if direction == 'left':
            points = [
                (screen_width//2 + arrow_size, screen_height//2 - arrow_size//2),
                (screen_width//2 - arrow_size, screen_height//2),
                (screen_width//2 + arrow_size, screen_height//2 + arrow_size//2)
            ]
            pygame.draw.polygon(self.screen, self.WHITE, points)
        elif direction == 'right':
            points = [
                (screen_width//2 - arrow_size, screen_height//2 - arrow_size//2),
                (screen_width//2 + arrow_size, screen_height//2),
                (screen_width//2 - arrow_size, screen_height//2 + arrow_size//2)
            ]
            pygame.draw.polygon(self.screen, self.WHITE, points)

    def draw_cross(self):
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        size = 100
        thickness = 20

        # Draw horizontal line
        pygame.draw.rect(self.screen, self.RED,
                         (screen_width//2 - size, screen_height//2 - thickness//2,
                          size*2, thickness))
        # Draw vertical line
        pygame.draw.rect(self.screen, self.RED,
                         (screen_width//2 - thickness//2, screen_height//2 - size,
                          thickness, size*2))

    def run_stimulation(self, duration_minutes=5):
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        while time.time() < end_time:
            # Randomly select a pattern
            current_pattern = random.choice(self.patterns)

            # Write stimulation onset marker
            self.write_stimulation_tag(time.time() - start_time, f"{current_pattern}_start")

            # Show the stimulation
            pattern_start = time.time()
            while time.time() - pattern_start < self.stim_duration:
                self.screen.fill(self.BLACK)

                if current_pattern == 'left':
                    self.draw_arrow('left')
                elif current_pattern == 'right':
                    self.draw_arrow('right')
                elif current_pattern == 'stop':
                    self.draw_cross()
                # For 'rest', just show black screen

                pygame.display.flip()

                # Check for quit event
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        # Write end marker before quitting
                        self.write_stimulation_tag(time.time() - start_time, "session_end")
                        pygame.quit()
                        return

            # Write stimulation end marker
            self.write_stimulation_tag(time.time() - start_time, f"{current_pattern}_end")

            # Rest period
            rest_start = time.time()
            self.write_stimulation_tag(rest_start - start_time, "rest_start")

            while time.time() - rest_start < self.rest_duration:
                self.screen.fill(self.BLACK)
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        self.write_stimulation_tag(time.time() - start_time, "session_end")
                        pygame.quit()
                        return

            self.write_stimulation_tag(time.time() - start_time, "rest_end")

if __name__ == "__main__":
    stim = StimulationDisplay()
    # Run stimulation for 5 minutes
    stim.run_stimulation(5)
