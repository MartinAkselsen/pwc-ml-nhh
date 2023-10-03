
# Data cleaning and EDA ---------------------------------------------------

# Load packages and custom functions
library(tidyverse)
library(tidymodels)
library(readxl)
library(xgboost)

# Init - run the code below -----------------------------------------------

# Load data from the "input"-folder (if this doesnt work, check if you are in an
# RStudio project and that you have the folder input)
houses_raw      <- read_excel("input/houses.xlsx")
geo_raw         <- read_excel("input/geo.xlsx")
zip_raw         <- read_excel("input/zip.xlsx")
income_raw      <- read_excel("input/income.xlsx")
attributes_raw  <- read_excel("input/attributes.xlsx")

geo <- geo_raw |>  
  select(id, kommune_no, kommune_name, fylke_no, fylke_name)

income <- income_raw |>  
  select(
    zip_no = postnr, 
    avg_income = gjsnitt_inntekt, 
    avg_fortune = gjsnitt_formue
  )

houses <- houses_raw |>  
  left_join(geo,     by = "id") |>  
  left_join(zip_raw, by = "id") |>  
  left_join(income,  by = "zip_no") |>  
  left_join(attributes_raw, by = "id")

houses <- houses |>  
  mutate(debt = replace_na(debt, 0),
         expense = replace_na(expense, 0),
         tot_price = price + debt, 
         tot_price_per_sqm = tot_price / sqm,
         log_tot_price = log(tot_price)) |> 
  drop_na()

# Check distribution of total price
houses |>  
  ggplot(aes(x = tot_price)) +
  geom_histogram(fill = "dodgerblue3", color = "white") +
  scale_x_continuous(labels = scales::number) +
  labs(x = "Price",
       y = "Count") +
  theme_minimal()

# Distribution after log-transform
houses |>  
  ggplot(aes(x = tot_price)) +
  geom_histogram(fill = "dodgerblue3", color = "white") +
  scale_x_log10(labels = scales::number) +
  labs(x = "Price, log axis",
       y = "Count") +
  theme_minimal()

# Simple scatter plot of debt vs price
houses |> 
  ggplot(aes(x = debt, y = price)) +
  geom_point()


# Exercises: Data cleaning and EDA ----------------------------------------

#  1.  Plot the distribution (histogram) of `avg_income` and `avg_fortune`. Are there any outliers?
#  2.  Explore the relationship between `sqm` and `sqm_use` using a scatter plot. What do you see?
#  3.  Find out which year the largest quantity of listed houses were built. 
#  4.  Study the relationship between `tot_price` and `sqm` using a scatter plot with a trend line 
#      (hint: use `geom_smooth`)


# Linear models -----------------------------------------------------------


## Splitting the data
set.seed(42)

split <- initial_split(houses, prop = 3/4)
train <- training(split)
test  <- testing(split)

str_glue("We have {nrow(train)} obs in training and {nrow(test)} obs for testing")

## Creating a recipe
lm_recipe <- train |>  
  recipe(log_tot_price ~ sqm + kommune_name + expense) |>  
  step_other(kommune_name, threshold = 500) |> 
  step_log(sqm, expense, offset = 1) |>  
  prep()

train_prepped <- bake(lm_recipe, train)
test_prepped  <- bake(lm_recipe, test)

## Define your model (choice of algorithm)
lm_spec <- linear_reg() |>  
  set_engine("lm")

# Combine recipe with model
lm_wflow <- workflow(lm_recipe, lm_spec)
lm_model <- fit(lm_wflow, train)

# Predict with your model
model_preds <- predict(lm_model, test)

# Evaluate the model
# Define function
evaluate_model <- function(model, eval_data) {
  model_preds <- 
    predict(model, eval_data) %>% 
    bind_cols(eval_data) %>% 
    rename(estimate     = .pred,
           truth        = tot_price) %>% 
    mutate(estimate     = exp(estimate),
           abs_dev      = abs(truth - estimate),
           abs_dev_perc = abs_dev/truth)
  
  mape(model_preds, truth, estimate)
}

evaluate_model(lm_model, test)

# Remember, we generally want a low MAPE-value

# Exercises, linear models ------------------------------------------------

  
#  1.  Examine how the model is affected when using `sqm_use` instead of `sqm` in the model. 
#      Do you see an issue using *both* of these variables in the model?
#  2.  Add `avg_income` and `avg_fortune` to your model. Does this improve your results?
#  3.  Change the engine to "glm" and set family to "Gamma". How does this affect your results?
  

# XGBoost, run the code below -------------------------------------------------------------

## We can use XGBoost to create a better model
xg_recipe <- train |>  
  recipe(log_tot_price ~ sqm + expense + kommune_name + lat + lng) |>  
  step_other(kommune_name, threshold = 200) |>  
  step_integer(kommune_name)

# We here choose "boost_tree" and set its hyperparameters
model_spec <- boost_tree(trees = 350, tree_depth = 6) |>  
  set_engine("xgboost") |>  
  set_mode("regression")

xg_wflow <- workflow(xg_recipe, model_spec)
xg_model <- fit(xg_wflow, train)

evaluate_model(xg_model, test)


# Exercises, xgboost ------------------------------------------------------

#  1.  Set `tree_depth` to 8 and re-train your model. Investigate how this affects your model results.
#  2.  We usually evaluate the model on the test set, but performance on the train set can also be interesting. 
#      Set `learn_rate` to 0.99 (a very high value) and evaluate your model prediction on the *training set*. 
#      What happens, and why? What happens with the performance on the test set?
#  3.  Add more relevant variables from `houses` to your xgboost-model. Does this improve your results? 
#      Hint: You might have to change some `recipe`-steps as well, to preprocess the new variables. 
#      `step_dummy(all_nominal_predictors())` could be used if you add more nominal variables to the model.
#  4.  For linear models, using latitude and longitude directly in the model makes little sense
#       , but for XGBoost, this can work well. Discuss why.