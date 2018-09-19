from maze import Maze
import turtle
import sys
from robot import Robot
import time

class ShowRobot(object):
    '''
    Creates a Turtle maze object to display robot exploration during testing
    the robot agent.
    '''
    def __init__(self, test_maze):
        '''
        Takes in test_maze.txt information to create and display maze
        
        test_maze: file path for maze dimensional information (string)
        '''
        # Intialize the maze dimensions from txt file.
        # Maze is centered on (0,0), squares are 20 units in length.
        self.test_maze = Maze(test_maze)
        self.sq_size = 20
        self.origin = self.test_maze.dim * self.sq_size / -2

        # Intialize the window and drawing turtle.
        self.window = turtle.Screen()
        self.env = turtle.Turtle()
        self.env.speed(0)
        self.env.hideturtle()
        self.env.penup()
        
        
    def start_maze(self):
        '''
        Creates the baseline maze envirnment from the information within the
        given .txt tester maze file.
        
        This env does not yet include the robot agent's path.
        '''
        # Iterate through squares one by one to decide where to draw walls.
        for x in range(self.test_maze.dim):
            for y in range(self.test_maze.dim):
                if not self.test_maze.is_permissible([x,y], 'up'):
                    self.env.goto(self.origin + self.sq_size * x,
                                  self.origin + self.sq_size * (y+1))
                    self.env.setheading(0)
                    self.env.pendown()
                    self.env.forward(self.sq_size)
                    self.env.penup()
    
                if not self.test_maze.is_permissible([x,y], 'right'):
                    self.env.goto(self.origin + self.sq_size * (x+1),
                                  self.origin + self.sq_size * y)
                    self.env.setheading(90)
                    self.env.pendown()
                    self.env.forward(self.sq_size)
                    self.env.penup()
    
                # Only check bottom wall if on lowest row.
                if y == 0 and not self.test_maze.is_permissible([x,y], 'down'):
                    self.env.goto(self.origin + self.sq_size * x,
                               self.origin)
                    self.env.setheading(0)
                    self.env.pendown()
                    self.env.forward(self.sq_size)
                    self.env.penup()
    
                # Only check left wall if on leftmost column.
                if x == 0 and not self.test_maze.is_permissible([x,y], 'left'):
                    self.env.goto(self.origin,
                               self.origin + self.sq_size * y)
                    self.env.setheading(90)
                    self.env.pendown()
                    self.env.forward(self.sq_size)
                    self.env.penup()
                    
                    
    def draw_robot_action(self, loc):
        '''
        Creates a square fill for every environment position explored by robot
        agent and draws it onto the Turtle maze environment display.
        '''
        self.env.goto(self.origin + loc[0] * self.sq_size + 0.75, 
                      self.origin + loc[1] * self.sq_size + 0.75)
        self.env.setheading(90)
        self.env.begin_fill()
        self.env.pendown()
        self.env.pencolor('green')
        self.env.fillcolor('green')
        self.env.forward(18)
        self.env.right(90)
        self.env.forward(18)
        self.env.right(90)
        self.env.forward(18)
        self.env.right(90)
        self.env.forward(18)
        self.env.penup()
        self.env.end_fill()
        
        time.sleep(0.00001)

        
if __name__ == '__main__':
    '''
    This function uses Python's turtle library to draw a picture of the maze
    given as an argument when running the script.
    '''
    display = ShowRobot('test_maze_02.txt')
    display.start_maze()

    display.window.exitonclick()