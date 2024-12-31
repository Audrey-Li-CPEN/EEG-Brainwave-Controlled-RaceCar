import pygame
import random
import time
from pygame.locals import *
import csv
import os
from datetime import datetime

# Idea for this stimulation script for Motor Imagery task data recording on user's computer
# This script will generate a stimulation pattern on the screen and record the data in a CSV file
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
        
        # Sampling and timing parameters
        self.sampling_rate = 128  # Hz
        self.sample_interval = 1.0 / self.sampling_rate
        
        # CSV handling
        self.csv_path = self.generate_csv_path()
        self.last_write_time = 0
        
    def generate_csv_path(self):
        # Monitor the OpenViBE output directory for new CSV files
        openvibe_output_dir = "path/to/openvibe/output"  # !! Need to update this path depend on different csv output dir
        while True:
            csv_files = [f for f in os.listdir(openvibe_output_dir) if f.endswith('.csv')]
            if csv_files:
                # Get the most recent CSV file
                latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(openvibe_output_dir, x)))
                return os.path.join(openvibe_output_dir, latest_csv)
            time.sleep(0.1)  # Wait for OpenViBE to create the CSV

    def write_stimulation_label(self, timestamp, pattern):
        try:
            # Read existing CSV content
            with open(self.csv_path, 'r') as file:
                lines = list(csv.reader(file))
                
            # Find the row with the closest timestamp
            for i, row in enumerate(lines):
                if row and float(row[0]) >= timestamp:
                    # Write the stimulation label to column 'T' (assuming it's the 20th column)
                    # Modify the column index based on your CSV structure
                    lines[i][19] = pattern
                    break
            
            # Write back the modified content
            with open(self.csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(lines)
                
        except Exception as e:
            print(f"Error writing to CSV: {e}")

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
            self.write_stimulation_label(time.time() - start_time, f"{current_pattern}_start")
            
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
                        self.write_stimulation_label(time.time() - start_time, "session_end")
                        pygame.quit()
                        return
            
            # Write stimulation end marker
            self.write_stimulation_label(time.time() - start_time, f"{current_pattern}_end")
            
            # Rest period
            rest_start = time.time()
            self.write_stimulation_label(rest_start - start_time, "rest_start")
            
            while time.time() - rest_start < self.rest_duration:
                self.screen.fill(self.BLACK)
                pygame.display.flip()
                
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        self.write_stimulation_label(time.time() - start_time, "session_end")
                        pygame.quit()
                        return
            
            self.write_stimulation_label(time.time() - start_time, "rest_end")

if __name__ == "__main__":
    stim = StimulationDisplay()
    # Run stimulation for 5 minutes
    stim.run_stimulation(5)
