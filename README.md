# algae_farm_localization

Monte Carlo localization of an [two-thruster underwater vehicle](https://github.com/smarc-project/smarc_stonefish_sims/tree/noetic-devel/sam_stonefish_sim) 
in a [simulated algae farm environment](https://github.com/smarc-project/smarc_stonefish_sims) consisting of 6 landmarks.

The motion model is similar to that in 
[dead-reckoning in smarc_navigation](https://github.com/smarc-project/smarc_navigation/tree/noetic-devel/sam_dead_reckoning),
where the sum of the two thruster RPMs are used for translational velocity. The measurement model uses range observation of the landmarks from sidescan sonar.

Some experimental results are found in the [experiment folder](https://github.com/luxiya01/algae_farm_localization/tree/main/experiments).
