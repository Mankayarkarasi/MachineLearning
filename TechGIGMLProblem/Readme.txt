Problem description: 

Given a sequence of CLICK events performed by User during a typical session, the goal is to predict whether the user is likely to make a purchase or drop out of the path.

Each record/line in the file is an action or a click done by a user in that session, each session is a unique user. They will have multiple records in a session based on the actions and clicks they are doing to complete an order or add to a checkout or browsing and exiting without checkout or order.

The funnel name will give info of how the customers traverse through the website , Customer start with upper funnel where they do learn and do product verification and then they go to middle where they select products and then finish up in lower with payment and other address info.


The train data consist of :

id - Unique Session Id of the user
Ad - If the user has clicked on "Ad" then the link is given else null value
link - link id which the user has clicked for interaction
timestamp - timestamp of the action performed
checkout - has the user performed any checkout during the session 0 is no checkout and 1 is checkedout sucessfully
order_placed - the user has made the payment and provided the address details
grp  -group id which contains similar links performed by the user
funnel_level :
upper -  has browsed various products and the user is intrested in learning about the product and does product verification 
middle - does product selection and add to cart
lower - make payment and provide address info

Models to analyze the followings:

Key differentiating sequences/sub-sequences/clicks between three groups
Factors lead to drop out of users & where they drop out
Presentation of the approach used to develop the framework
Which should describes hypothesis formulation and data usages, Feature creation and selection, Intervention strategy for group 2 and 3 and Model and technique used.


Evaluation Criteria:


Hypothesis formulation and data usages
Feature creation and selection
Intervention strategy for group 2 and 3
Model and technique used
Accuracy of prediction

The challenge is to submit the output file for test data in the same format of "sample-submission.csv". 


