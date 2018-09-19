import numpy as np
import random

class Robot(object):
    def __init__(self, maze_dim):
        """
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        """
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.location = [maze_dim - 1, 0]
        self.goal_area = [self.maze_dim/2 - 1, self.maze_dim/2]
        self.dir_grid = [[0 for row in range(0, self.maze_dim)] for col in range(0, self.maze_dim)]
        self.count_grid = [[0 for row in range(1, self.maze_dim + 1)] for col in range(1, self.maze_dim + 1)]
        self.action_grid = [[0 for row in range(1, self.maze_dim + 1)] for col in range(1, self.maze_dim + 1)]
        self.model = [[0 for row in range(0, self.maze_dim)] for col in range(0, self.maze_dim)]
        self.cell_count = 0
        self.action_count = 0
        self.goal_success = False
        self.training = False
        random.seed(0)

    def reset(self):
        """
        Resets the Robot class attributes to the initialized values, for most of the variables.

        :return: NULL
        """
        self.location = [self.maze_dim - 1, 0]
        self.heading = 'up'
        self.training = not self.training
        print('\n\nResetting robot for Training')
        print('\nCount Grid:\n{}'.format(self.count_grid))
        print('\nDirection Grid:\n{}'.format(self.dir_grid))
        print('\nAction Grid:\n{}'.format(self.action_grid))

    def map_cell(self, sensors):
        """
        Records the directional values for each given cell based on the sensor
        input data.

        :param sensors: the sensor values of agent for a given cell 
            (a list of ints, i.e. [0, 0, 1])
        
        :return: NULL
        """
        x, y = self.location
        headings = ['left', 'up', 'right', 'down']
        directions = [8, 1, 2, 4]
        
        if self.dir_grid[x][y] == 0:
            for i in range(len(headings)):
                if self.heading == headings[i]:
                    self.dir_grid[x][y] += directions[(i + 2) % 4]
                    if sensors[0] > 0:
                        self.dir_grid[x][y] += directions[i - 1]
                    if sensors[1] > 0:
                        self.dir_grid[x][y] += directions[i]
                    if sensors[2] > 0:
                        self.dir_grid[x][y] += directions[(i + 1) % 4]
        
        self.dir_grid[self.maze_dim - 1][0] = 1
    
    def breadcrumb(self):
        """
        Counts the number of unique cells visited.

        :param: NULL
        
        :return: NULL
        """
        x, y = self.location
        
        if self.count_grid[x][y] == 0:
            self.count_grid[x][y] = 1
            self.cell_count += 1
    
    def update_model(self, location, tune):
        """
        Updates the values within the model.
        
        :param location: the model agent location within the map grid
            (a tuple of ints, i.e. [0, 1])
            
        :param tune: Model tuning variable
            (a int, i.e. 1)
        
        :return: NULL
        """
        x, y = location
        self.model[x][y] = tune

    def act_legal(self, location):
        """
        Determines what actions are legal and returns the value
        
        :param location: Location of agent model
            (a tuple of ints, i.e. (0, 1))
        
        :return actions: legal actions available
            (a list of strings, i.e. ['left', 'up'])
        """
        x, y = location
        actions = []
        
        vals = [[1, 3, 5, 7, 9, 11, 13, 15],
                [2, 3, 6, 7, 10, 11, 14, 15],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [8, 9, 10, 11, 12, 13, 14, 15]]
                
        possible = ['up', 'right', 'down', 'left']
        
        for i in range(len(vals)):
            if self.dir_grid[x][y] in vals[0]:
                actions.extend([possible[0]])
            if self.dir_grid[x][y] in vals[1]:
                actions.extend([possible[1]])
            if self.dir_grid[x][y] in vals[2]:
                actions.extend([possible[2]])
            if self.dir_grid[x][y] in vals[3]:
                actions.extend([possible[3]])
            
            return actions
            
    def make_model(self):
        """
        Creates ML model for agent based on the first training run recorded
        sensor data and cell information

        :param: NULL
        
        :return: NULL
        """
        opened = []
        tune = 1
        trans = [[-1, 0], [0, 1], [1, 0], [0, -1]]

        x = self.goal_area[0]
        y = self.goal_area[1]

        # Store adjacent directional options based on current cell location
        opened.append(((x, y), tune))
        opened.append(((x + trans[2][0],    y + trans[2][1]), tune))
        opened.append(((x + trans[3][0],    y + trans[3][1]), tune))
        opened.append(((x + trans[2][0],    y + trans[3][1]), tune))
        
        # Update agent's model's cell values
        for cell in opened:
            self.update_model(cell[0], cell[1])

        # Check all valid possible directions for model agent
        while self.model[self.maze_dim -1][0] == 0 and len(opened) != 0:
            location, tune = opened.pop(0)
            actions = self.act_legal(location)

            if 'up' in actions and self.count_grid[location[0]][location[1]] != 0:
                x2 = location[0] + trans[0][0]
                y2 = location[1] + trans[0][1]
                new_location = x2, y2
                if self.model[x2][y2] == 0 and self.count_grid[location[0]][location[1]] != 0:
                    opened.append((new_location, tune + 1))
                    self.update_model(new_location, tune + 1)
            
            if 'right' in actions:
                x2 = location[0] + trans[1][0]
                y2 = location[1] + trans[1][1]
                new_location = x2, y2
                if self.model[x2][y2] == 0 and self.count_grid[location[0]][location[1]] != 0:
                    opened.append((new_location, tune + 1))
                    self.update_model(new_location, tune + 1)
            
            if 'down' in actions:
                x2 = location[0] + trans[2][0]
                y2 = location[1] + trans[2][1]
                new_location = x2, y2
                if self.model[x2][y2] == 0 and self.count_grid[location[0]][location[1]] != 0:
                    opened.append((new_location, tune + 1))
                    self.update_model(new_location, tune + 1)
        
            if 'left' in actions:
                x2 = location[0] + trans[3][0]
                y2 = location[1] + trans[3][1]
                new_location = x2, y2
                if self.model[x2][y2] == 0:
                    opened.append((new_location, tune + 1))
                    self.update_model(new_location, tune + 1)
    
    def make_action_grid(self):
        """
        Determines and stores the best action for the agent for each cell.
        
        :param: NULL
        
        :return: NULL
        """
        vals = [[1, 3, 5, 7, 9, 11, 13, 15],
                [2, 3, 6, 7, 10, 11, 14, 15],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [8, 9, 10, 11, 12, 13, 14, 15]]
                
        possible = ['up', 'right', 'down', 'left']
        
        trans = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        
        for i in range(len(self.model)):
            for j in range(len(self.model[0])):
                for k in range(len(vals)):
                    if self.dir_grid[i][j] in vals[k]:
                        if self.model[i + trans[k][0]][j + trans[k][1]] == self.model[i][j] - 1:
                            self.action_grid[i][j] = possible[k]

    def make_action(self, sensors):
        """
        Determines the rotation and movement based on what the best action for the robot to execute.

        :param sensors: the sensor values of agent for a given cell
            (a list of ints, i.e. [0, 0, 1])

        :return: rotation, movement: the rotation and movement the robot agent should act on next
            (a tuple of ints, i.e. [90, 1])
        """
        x, y = self.location
        moves = ['left', 'forward', 'right']
        possible_moves = []
        rotation = 0
        movement = 0

        # Store possible moves based on sensor data
        for i in range(len(sensors)):
            # There is a wall
            if sensors[i] > 0:
                possible_moves.extend([moves[i]])
        # No possible moves, robot agent hit dead end
        if not possible_moves:
            rotation = 90
            movement = 0

        moves_up = []
        moves_down = []
        moves_left = []
        moves_right = []
        actions = []

        # EXPLORATION
        # Store move based on heading and sensors
        if not self.training:
            if self.heading == 'up':
                if 'right' in possible_moves:
                    moves_right.extend(range(1, sensors[2] + 1))
                if 'forward' in possible_moves:
                    moves_up.extend(range(1, sensors[1] + 1))
                if 'left' in possible_moves:
                    moves_left.extend(range(1, sensors[0] + 1))
            elif self.heading == 'right':
                if 'right' in possible_moves:
                    moves_down.extend(range(1, sensors[2] + 1))
                if 'forward' in possible_moves:
                    moves_right.extend(range(1, sensors[1] + 1))
                if 'left' in possible_moves:
                    moves_up.extend(range(1, sensors[0] + 1))
            elif self.heading == 'down':
                if 'right' in possible_moves:
                    moves_left.extend(range(1, sensors[2] + 1))
                if 'forward' in possible_moves:
                    moves_down.extend(range(0, sensors[1] + 1))
                if 'left' in possible_moves:
                    moves_right.extend(range(1, sensors[0] + 1))
            elif self.heading == 'left':
                if 'right' in possible_moves:
                    moves_up.extend(range(1, sensors[2] + 1))
                if 'forward' in possible_moves:
                    moves_left.extend(range(1, sensors[1] + 1))
                if 'left' in possible_moves:
                    moves_down.extend(range(1, sensors[0] + 1))

            # Store actions based on movement options
            if 1 in moves_up:
                if self.count_grid[x - 1][y] != 1:
                    actions.extend([1])
            if 2 in moves_up:
                if self.count_grid[x - 2][y] != 1:
                    actions.extend([11])
            if 3 in moves_up:
                if self.count_grid[x - 3][y] != 1:
                    actions.extend([101])
            if 1 in moves_right:
                if self.count_grid[x][y + 1] != 1:
                    actions.extend([2])
            if 2 in moves_right:
                if self.count_grid[x][y + 2] != 1:
                    actions.extend([12])
            if 3 in moves_right:
                if self.count_grid[x][y + 3] != 1:
                    actions.extend([102])
            if 1 in moves_down:
                if self.count_grid[x + 1][y] != 1:
                    actions.extend([3])
            if 2 in moves_down:
                if self.count_grid[x + 2][y] != 1:
                    actions.extend([13])
            if 3 in moves_down:
                if self.count_grid[x + 3][y] != 1:
                    actions.extend([103])
            if 1 in moves_left:
                if self.count_grid[x][y - 1] != 1:
                    actions.extend([4])
            if 2 in moves_left:
                if self.count_grid[x][y - 2] != 1:
                    actions.extend([14])
            if 3 in moves_left:
                if self.count_grid[x][y - 3] != 1:
                    actions.extend([104])

            # Make sure there are valid actions available
            if actions:
                action = random.choice(actions)
                possible_actions = [1, 2, 3, 4, 11, 12, 13, 14, 101, 102, 103, 104]
                directions = ['up', 'right', 'down', 'left']

                for i in range(len(possible_actions)):
                    if possible_actions[i] == action:
                        movement = len(str(possible_actions[i]))
                        direction = directions[i % 4]
                    if x in self.goal_area and y in self.goal_area:
                        movement = 1
                    if self.action_count < 5:
                        movement = 1

                # Determine rotation value based on direction and heading
                for i in range(len(directions)):
                    if self.heading == directions[i]:
                        if direction == directions[i]:
                            rotation = 0
                        elif direction == directions[i - 1]:
                            rotation = -90
                        elif direction == directions[(i + 1) % 4]:
                            rotation = 90
            # Determine movement value based on rotation and possible moves
            elif possible_moves != 0:
                rotations = [-90, 0, 90]
                possible_rotations = []
                for i in range(len(sensors)):
                    for j in range(len(possible_moves)):
                        if possible_moves[j] == moves[i]:
                            possible_rotations.append(rotations[i])
                            rotation = random.choice(possible_rotations)
                            movement = 1
            # Robot agent has hit a dead end, turn around
            else:
                movement = 0
                rotation = 90

        # TRAINING
        # Determine movement based on robot agent trained model
        if self.training:
            directions = ['up', 'right', 'down', 'left']
            delta = [[-1, 0], [0, 1], [1, 0], [0, -1]]
            action = self.action_grid[x][y]

            for i in range(len(directions)):
                # Determine movement value, 1, 2, 3
                if self.action_grid[x][y] == directions[i]:
                    if self.action_grid[x + delta[i][0]][y + delta[i][1]] == directions[i]:
                        if self.action_grid[x + (2 * delta[i][0])][y + (2 * delta[i][1])] == directions[i]:
                            movement = 3
                        else:
                            movement = 2
                    else:
                        movement = 1
                # Determine rotation value, -90, 0, 90
                if self.heading == directions[i]:
                    if action == directions[i]:
                        rotation = 0
                    elif action == directions[i - 1]:
                        rotation = -90
                    elif action == directions[(i + 1) % 4]:
                        rotation = 90

        # Determine new heading based on current heading and rotation values
        new_heading = ''
        if self.heading == 'up':
            if rotation == 0:
                new_heading = 'up'
                x -= movement
            elif rotation == -90:
                new_heading = 'left'
                y -= movement
            elif rotation == 90:
                new_heading = 'right'
                y += movement
        elif self.heading == 'right':
            if rotation == 0:
                new_heading = 'right'
                y += movement
            elif rotation == -90:
                new_heading = 'up'
                x -= movement
            elif rotation == 90:
                new_heading = 'down'
                x += movement
        elif self.heading == 'down':
            if rotation == 0:
                new_heading = 'down'
                x += movement
            elif rotation == -90:
                new_heading = 'right'
                y += movement
            elif rotation == 90:
                new_heading = 'left'
                y -= movement
        else:
            if rotation == 0:
                new_heading = 'left'
                y -= movement
            elif rotation == -90:
                new_heading = 'down'
                x += movement
            elif rotation == 90:
                new_heading = 'up'
                x -= movement

        # Update robot agent class variables based on action
        self.heading = new_heading
        self.location = [x, y]
        self.action_count += 1

        return rotation, movement

    def next_move(self, sensors):
        """
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.

        :param sensors: the sensor values of agent for a given cell
            (a list of ints, i.e. [0, 0, 1])

        :return: rotation, movement: the rotation and movement the robot agent should act on next
                 resetting the agent takes a value of ('reset', 'reset')
            (a tuple of ints, i.e. [90, 1])
        """
        rotation = 0
        movement = 0
        # Record agent sensor data current location cell
        x, y = self.location
        self.map_cell(sensors)
        self.breadcrumb()

        # Check if robot agent is within goal area
        if [x, y] == self.goal_area:
            self.goal_success = True
            print('Successfully found goal. Agent at {}, {}.'.format(x, y))

        # Reset run
        if not self.training and self.goal_success:
            if (self.cell_count >= (self.maze_dim ** 2)) or (self.action_count >= 300):
                # Training run results
                print('\nTraining Run Results:\n')
                print('{}'.format(self.dir_grid))
                    
                # Make model
                self.make_model()
                print('\nTraining Model:\n')
                print('{}'.format(self.model))

                # Make action grid
                self.make_action_grid()
                self.reset()
                return 'Reset', 'Reset'

        # Final run
        rotation, movement = self.make_action(sensors)
        print('\nLocation: {} \tRotation: {} \tMovement: {} \tAction Count:{}'.format(self.location,
                                                                                    rotation,
                                                                                    movement,
                                                                                    self.action_count))
        return rotation, movement
