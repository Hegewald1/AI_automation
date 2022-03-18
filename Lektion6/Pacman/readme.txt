Aarhus, EAAA. April 2020.

You are NOT allowed to distribute these files.

People who are interested in these python Pacman files
MUST download themselves from the Berkeley site.


To get started, run Gridworld in manual control mode, which uses the arrow keys
python gridworld.py -m


With the Q-learning update in place, you can watch your Q-learner learn under manual control, using the keyboard:
python gridworld.py -a q -k 5 -m

Time to play some Pacman! Pacman will play games in two phases. In the first phase, training,
 Pacman will begin to learn about the values of positions and actions. 
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid

