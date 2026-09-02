import cv2
import numpy as np
import math
import random

class Robot:
    def __init__(self, robot_id, x, y, heading=0):
        self.id = robot_id
        self.x = float(x)
        self.y = float(y)
        self.heading = heading # in radians
        self.step_size = 3.0
        self.turn_size = 0.15

    def forward(self):
        self.x += self.step_size * math.cos(self.heading)
        self.y += self.step_size * math.sin(self.heading)

    def backward(self):
        self.x -= self.step_size * math.cos(self.heading)
        self.y -= self.step_size * math.sin(self.heading)

    def left(self):
        self.heading -= self.turn_size

    def right(self):
        self.heading += self.turn_size

class SwarmSimulation:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.robots = {}
        self.polygon = []
        self.state = "DEFINE_POLYGON" 
        self.window_name = "Swarm Mesh Simulation"
        self.robot_counter = 0
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_event)
        
    def _mouse_event(self, event, x, y, flags, param):
        # Click to drop polygon vertices during the initialization phase
        if self.state == "DEFINE_POLYGON" and event == cv2.EVENT_LBUTTONDOWN:
            self.polygon.append((x, y))

    def add_robot(self, x, y):
        r_id = self.robot_counter
        self.robots[r_id] = Robot(r_id, x, y, heading=random.uniform(0, 2*math.pi))
        self.robot_counter += 1
        return r_id

    def delete_robot(self, robot_id=None):
        if not self.robots:
            return
        if robot_id is None or robot_id not in self.robots:
            # Delete the most recently added robot if no specific ID is given
            robot_id = list(self.robots.keys())[-1]
        del self.robots[robot_id]

    def render(self):
        # Initialize a blank black canvas
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw the target Polygon
        if len(self.polygon) > 0:
            pts = np.array(self.polygon, np.int32).reshape((-1, 1, 2))
            is_closed = (self.state == "SIMULATE")
            cv2.polylines(frame, [pts], is_closed, (255, 0, 0), 2)
            for pt in self.polygon:
                cv2.circle(frame, pt, 3, (0, 0, 255), -1)
            
        # Render Robots
        for r_id, robot in self.robots.items():
            pt = (int(robot.x), int(robot.y))
            # Main robot dot
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            # Heading indicator (shows which way is 'forward')
            end_pt = (int(robot.x + 12 * math.cos(robot.heading)), 
                      int(robot.y + 12 * math.sin(robot.heading)))
            cv2.line(frame, pt, end_pt, (0, 200, 255), 2)
            
        # HUD overlays
        if self.state == "DEFINE_POLYGON":
            cv2.putText(frame, "Click to map polygon. Press 'SPACE' to confirm & start.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Press 'a' to Add, 'd' to Delete, 'ESC' to Quit", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        cv2.imshow(self.window_name, frame)

    def run(self):
        while True:
            self.render()
            
            if self.state == "SIMULATE":
                for r in self.robots.values():
                    # Placeholder random walk testing the movement functions
                    action = random.choice([r.forward, r.left, r.right])
                    action()
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27: # ESC key to exit
                break
            elif key == ord(' ') and self.state == "DEFINE_POLYGON":
                self.state = "SIMULATE"
            elif key == ord('a') and self.state == "SIMULATE":
                self.add_robot(random.randint(100, self.width-100), random.randint(100, self.height-100))
            elif key == ord('d') and self.state == "SIMULATE":
                self.delete_robot()
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sim = SwarmSimulation()
    sim.run()