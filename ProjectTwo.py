# Treasure Hunt Game Notebook

## Read and Review Your Starter Code
The theme of this project is a popular treasure hunt game in which the player needs to find the treasure before the pirate does. While you will not be developing the entire game, you will write the part of the game that represents the intelligent agent, which is a pirate in this case. The pirate will try to find the optimal path to the treasure using deep Q-learning. 

You have been provided with two Python classes and this notebook to help you with this assignment. The first class, TreasureMaze.py, represents the environment, which includes a maze object defined as a matrix. The second class, GameExperience.py, stores the episodes – that is, all the states that come in between the initial state and the terminal state. This is later used by the agent for learning by experience, called "exploration". This notebook shows how to play a game. Your task is to complete the deep Q-learning implementation for which a skeleton implementation has been provided. The code blocks you will need to complete has #TODO as a header.

First, read and review the next few code and instruction blocks to understand the code that you have been given.


from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
from TreasureMaze import TreasureMaze
from GameExperience import GameExperience
%matplotlib inline


The following code block contains an 8x8 matrix that will be used as a maze object:

maze = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
])

This helper function allows a visual representation of the maze object:

def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    pirate_row, pirate_col, _ = qmaze.state
    canvas[pirate_row, pirate_col] = 0.3   # pirate cell
    canvas[nrows-1, ncols-1] = 0.9 # treasure cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')

The pirate agent can move in four directions: left, right, up, and down. 

While the agent primarily learns by experience through exploitation, often, the agent can choose to explore the environment to find previously undiscovered paths. This is called "exploration" and is defined by epsilon. This value is typically a lower value such as 0.1, which means for every ten attempts, the agent will attempt to learn by experience nine times and will randomly explore a new path one time. You are encouraged to try various values for the exploration factor and see how the algorithm performs.


    return img

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


# Exploration factor
epsilon = 0.1

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)


The sample code block and output below show creating a maze object and performing one action (DOWN), which returns the reward. The resulting updated environment is visualized.

qmaze = TreasureMaze(maze)
canvas, reward, game_over = qmaze.act(DOWN)
print("reward=", reward)
show(qmaze)



def play_game(model, qmaze, pirate_cell):
    qmaze.reset(pirate_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False

This function helps you to determine whether the pirate can win any game at all. If your maze is not well designed, the pirate may not win any game at all. In this case, your training would not yield any result. The provided maze in this notebook ensures that there is a path to win and you can run this method to check.

def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True

The code you have been given in this block will build the neural network model. Review the code and note the number of layers, as well as the activation, optimizer, and loss functions that are used to train the model.

def build_model(maze):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model

# #TODO: Complete the Q-Training Algorithm Code Block

This is your deep Q-learning implementation. The goal of your deep Q-learning implementation is to find the best possible navigation sequence that results in reaching the treasure cell while maximizing the reward. In your implementation, you need to determine the optimal number of epochs to achieve a 100% win rate.

You will need to complete the section starting with #pseudocode. The pseudocode has been included for you.


def qtrain(model, maze, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 10)
    max_memory = opt.get('max_memory', 500)
    data_size = opt.get('data_size', 16)
    qmaze = TreasureMaze(maze)
    experience = GameExperience(model, max_memory=max_memory)

    print("Starting Q-Training...")
    for epoch in range(n_epoch):
        pirate_start = random.choice(qmaze.free_cells)
        qmaze.reset(pirate_start)
        game_over = False
        envstate = qmaze.observe()
        score = 0

        while not game_over:
            prev_envstate = envstate

            # Choose action based on exploration vs exploitation
            if np.random.rand() < epsilon:
                action = np.random.choice(qmaze.valid_actions())
            else:
                q_values = model.predict(envstate.reshape(1, -1))
                action = np.argmax(q_values[0])

            # Apply action, get rewards and new state
            envstate, reward, game_status = qmaze.act(action)
            score += reward

            # Store episode in Experience replay object
            episode = [prev_envstate, action, reward, envstate, game_status]
            experience.remember(episode)

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            model.train_on_batch(inputs, targets)

            # Check if the game is won
            if game_status == 'win':
                print(f"Win at Epoch {epoch + 1}")
                game_over = True

        # Log the progress of each epoch
        print(f"Epoch: {epoch + 1}, Score: {score}, Epsilon: {epsilon:.4f}")

        # Update epsilon to favor exploitation over time
        epsilon = max(0.01, epsilon * 0.98)

    print("Training complete!")



## Test Your Model

Now we will start testing the deep Q-learning implementation. To begin, select **Cell**, then **Run All** from the menu bar. This will run your notebook. As it runs, you should see output begin to appear beneath the next few cells. The code below creates an instance of TreasureMaze.

qmaze = TreasureMaze(maze)
show(qmaze)

In the next code block, you will build your model and train it using deep Q-learning. Note: This step takes several minutes to fully run.

model = build_model(maze)
qtrain(model, maze, n_epoch=50, max_memory=1000, data_size=50)


Starting Q-Training...
Win at Epoch 1
Epoch: 1, Score: 0.33999999999999986, Epsilon: 0.1000
Win at Epoch 2
Epoch: 2, Score: -31.96999999999999, Epsilon: 0.0980
Win at Epoch 3
Epoch: 3, Score: -13.45, Epsilon: 0.0960
Win at Epoch 4
Epoch: 4, Score: -31.00999999999908, Epsilon: 0.0941
Win at Epoch 5
Epoch: 5, Score: -31.100000000000023, Epsilon: 0.0922
Win at Epoch 6
Epoch: 6, Score: -15.789999999999985, Epsilon: 0.0904
Win at Epoch 7
Epoch: 7, Score: -31.15000000000014, Epsilon: 0.0886
Win at Epoch 8
Epoch: 8, Score: 0.8, Epsilon: 0.0868
Win at Epoch 9
Epoch: 9, Score: 0.88, Epsilon: 0.0851
Win at Epoch 10
Epoch: 10, Score: -13.109999999999985, Epsilon: 0.0834
Win at Epoch 11
Epoch: 11, Score: -2.6900000000000004, Epsilon: 0.0817
Win at Epoch 12
Epoch: 12, Score: -7.779999999999994, Epsilon: 0.0801
...
Epoch: 49, Score: 0.64, Epsilon: 0.0379
Win at Epoch 50
Epoch: 50, Score: -0.4200000000000004, Epsilon: 0.0372
Training complete!
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...


This cell will check to see if the model passes the completion check. Note: This could take several minutes.

completion_check(model, qmaze)
show(qmaze)


This cell will test your model for one game. It will start the pirate at the top-left corner and run play_game. The agent should find a path from the starting position to the target (treasure). The treasure is located in the bottom-right corner.

pirate_start = (0, 0)
play_game(model, qmaze, pirate_start)
show(qmaze)





