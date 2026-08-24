* __Programming Game AI by Example — Mat Buckland__
    * <font color=blue>more code-heavy</font>
----------

### Artificial Intelligence for Games — Ian Millington & John Funge
* <font color=blue>use as main book</font>
* fig 1.1. the AI model's sections
    * character AI
        * <font color=green>movement</font>
        * <font color=orange>decision making</font>
    * Group AI
        * strategy
        * to operate a group of characters
        * application: Chess, Risk games
* 1.2.1 <font color=green>movement</font>
    * Movement refers to algorithms that turn decisions into some kind of motion.
* 1.2.2 <font color=orange>decision making</font>
    * Decision making involves a character working out what to do next. Typically, each character has
a range of different behaviors that they could choose to perform.
* 1.2.3 <font color=brown>strategy</font>
    * In the context of this book, strategy refers to an overall approach used by a group of characters.
In this category are AI algorithms that don’t control just one character, but influence the behavior
of a whole set of characters. Each character in the group may (and usually will) have their own
decision making and movement algorithms, but overall their decision making will be influenced
by a group strategy.
* Requirement to build AI for games
    * movement requests must be turned to action using either animation, physics simulation
    * perception: info from env. fed to decision making system
    * managing amount of time & memory used by the AI modules

* Chapter 2 GAME AI
    * 2.1.3 perception window
        * I think it means that we have to design AI by taking into account the amount of player's interaction time with the AI player. Making a complex agent that needs minutes to reveal its intelligent behavior feels faulty/stupid to the player whose interaction with the agent is for a few seconds.
    * 2.2 the kind of AI in games
        * game AI consists of 
            * hacking: ad hoc solutions & neat effects
            * heuristics: rules of thumb that works most of the time
                * common heuristics
                    * most constrained
                    * do the most difficult thing first
                    * try the most promising thing first                    
            * algorithms
                * <font color=green>the focus of this book in on __algorithms__</font>
    * 2.3.1 processor issues
        * multi-thread programming may need special handling for some of the game consoles like ps3
        * virtual functions/indirection
            * while polymorphism offers code flexibility, sometimes it is factored out to get code speed up
    * 2.3.2 memory concerns
------------
#### Chapter 3 movement
* movement forms the lowest level of AI techniques in the model shown in fig 3.1
* AI & animation have a degree of overlap
* fig 3.2: the movement algorithm structure
* steering behaviors    
    * kinematics vs dynamics
        * kinematics is only concerned with moving an agent to a desired position.
        * A dynamic algorithm outputs forces or accelerations with the aim of changing the velocity of the character.
        * dynamics adds an extra layer of complexity        
    * page48
        *  ![img](hint-icon.jpg)__common approach for controlling cars in games__
            * engine's force & forces due to steering wheels are used to control the car motion.
            * This approach may not be suitable for control human characters.
        * most well-established steering algorithms work with acceleration outputs NOT forces.
    * page49
        * While simple, kinematic algorithms are widely used in games.
        * Dynamic algorithms are used less than kinematic methods.
    * section 3.3
        * steering behaviors may be
            * fundamental: just an atomic behavior            
            * combination of multiple fundamental behaviors
