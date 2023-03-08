# Personalized neural networks
The code in this repository is an attempt to apply personalization to a regression task using time-stamped data. It is not yet fully completed. 

A personalized model should be able to work "in general" and, when given specification information should return better predictions taking into account additional weights.
For a single price feature this approach might not be very effective.
However, if we add more features to data as, for example, the condition of the battery, which should affect the price of an item in a similar way, then the results could be quite interesting.

The advantage of personalized models is that they are quite effective even with a small sample of data due to the base in the form of a generic model.