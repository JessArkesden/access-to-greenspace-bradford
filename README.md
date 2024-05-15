**The full study can be found [here](markdown-version/access-to-greenspace-bradford.md) (markdown version).**

______________________________

### Summary

Greenspace is an important part of urban life, with benefits for humans and the environment. Analysing Bradford local authority district, this study builds on the growing literature addressing the challenge of greenspace accessibility by focusing on each deprivation dimension recorded in the 2021 Census.

Bradford is a large district, with a young and diverse population. Health and housing inequalities are known issues, whilst greenspace varies across the district in terms of quantity, type and size. The council has embedded greenspace within its long-term strategy, aiming to make sure they are safe, inclusive and that investment is delivered where it is needed most.

This study brings together 2021 Census data on each dimension of deprivation (education, employment, health and housing) by Output Area (OA), and combines this with Ordanance Survey data on greenspace in the district. After some initial data wrangling, a series of statistics and visualisations help to illustrate the data that will be used in the subsequent modelling.

First, using an OLS regression model, the variables are analysed to understand the significance of their relationship with access to greenspace - defined here as the distance to the nearest greenspace. This shows that employment deprivation is not statistically significant, and is subsequently removed.

Further investigation using a range of techniques, including Moran's I and LISA clusters, shows that there is spatial autocorrelation present and so a Geographically Weighted Regression model (GWR) is built. The results of this reject the null hypothesis that there is not a statistically significant relationship present between dimensions of deprivation and access to greenspace.

This study therefore argues that any future policy interventions need to take into consideration deprivation and its relationship with greenspace.

______________________________
### Disclaimer
This notebook was submitted for assessment within the *GEOG5402 Data Science for Urban Systems* module as part of the *MSc Urban Data Science & Analytics* programme at the *University of Leeds*.

**The content of this notebook is intended for educational and general use purposes only.**

Any use of the materials should adhere to the guidelines and policies of your educational institution.

**The author does not take any responsibility for how the materials in this repository are utilised.**
