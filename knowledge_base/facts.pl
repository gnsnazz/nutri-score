% SOGLIE NUTRIZIONALI

threshold(fat, low, 3.0).
threshold(fat, high, 20.0).
threshold(salt, high, 1.5).
threshold(fiber, good, 3.0).
threshold(protein, good, 5.0).
threshold(protein, high, 10.0).
threshold(sugar, high, 15.0).
threshold(sugar, very_high, 30.0).
threshold(fruit_veg, low, 40.0).
threshold(additives, max, 3).

isa(yogurt, dairy).
isa(milk, dairy).
isa(cheese, dairy).
isa(cream, dairy).

isa(bread, cereal).
isa(pasta, cereal).
isa(cereal, cereal).
isa(rice, cereal).

isa(juice, beverage).
isa(soda, beverage).
isa(cola, beverage).
isa(energy_drink, beverage).

isa(cookie, snack).
isa(biscuit, snack).
isa(chocolate, snack).
isa(chips, snack).
isa(candy, snack).

isa(sauce, condiment).
isa(ketchup, condiment).
isa(mayo, condiment).

isa(ham, processed_meat).
isa(sausage, processed_meat).
isa(salami, processed_meat).

is_a(X, Y) :- isa(X, Y).
is_a(X, Z) :- isa(X, Y), is_a(Y, Z).


expected_nutrient(dairy, protein, 3.0, 15.0).
expected_nutrient(dairy, sugar, 0.0, 8.0).
expected_nutrient(dairy, fat, 0.0, 10.0).

expected_nutrient(cereal, fiber, 3.0, 15.0).
expected_nutrient(cereal, sugar, 0.0, 10.0).
expected_nutrient(cereal, protein, 5.0, 15.0).

expected_nutrient(beverage, sugar, 0.0, 5.0).
expected_nutrient(beverage, protein, 0.0, 2.0).

expected_nutrient(snack, sugar, 0.0, 25.0).
expected_nutrient(snack, fat, 0.0, 30.0).

expected_nutrient(condiment, salt, 0.0, 2.0).
expected_nutrient(condiment, sugar, 0.0, 15.0).

expected_nutrient(processed_meat, salt, 0.0, 2.5).
expected_nutrient(processed_meat, protein, 10.0, 25.0).

inherits_expectation(Category, Nutrient, Min, Max) :-
    expected_nutrient(Category, Nutrient, Min, Max).

inherits_expectation(Category, Nutrient, Min, Max) :-
    is_a(Category, ParentCategory),
    expected_nutrient(ParentCategory, Nutrient, Min, Max).

keyword_category(yogurt, yogurt).
keyword_category(yoghurt, yogurt).
keyword_category(yaourt, yogurt).
keyword_category(skyr, yogurt).
keyword_category(milk, milk).
keyword_category(latte, milk).
keyword_category(cheese, cheese).
keyword_category(formaggio, cheese).
keyword_category(cream, cream).
keyword_category(creme, cream).

keyword_category(bread, bread).
keyword_category(pane, bread).
keyword_category(pasta, pasta).
keyword_category(cereal, cereal).
keyword_category(cereali, cereal).
keyword_category(rice, rice).
keyword_category(riso, rice).

keyword_category(juice, juice).
keyword_category(succo, juice).
keyword_category(soda, soda).
keyword_category(cola, cola).
keyword_category(drink, energy_drink).
keyword_category(energy, energy_drink).

keyword_category(cookie, cookie).
keyword_category(cookies, cookie).
keyword_category(biscuit, cookie).
keyword_category(biscotti, cookie).
keyword_category(chocolate, chocolate).
keyword_category(cioccolato, chocolate).
keyword_category(choco, chocolate).
keyword_category(chips, chips).
keyword_category(patatine, chips).
keyword_category(candy, candy).
keyword_category(caramelle, candy).

keyword_category(sauce, sauce).
keyword_category(salsa, sauce).
keyword_category(ketchup, ketchup).
keyword_category(mayo, mayo).
keyword_category(maionese, mayo).

keyword_category(ham, ham).
keyword_category(prosciutto, ham).
keyword_category(sausage, sausage).
keyword_category(salsiccia, sausage).
keyword_category(salami, salami).

marketing_keyword('bio').
marketing_keyword('organic').
marketing_keyword('vegan').
marketing_keyword('light').
marketing_keyword('fit').
marketing_keyword('natural').
marketing_keyword('detox').
marketing_keyword('sport').
marketing_keyword('protein').
marketing_keyword('wellness').
marketing_keyword('gluten free').
marketing_keyword('zero').
marketing_keyword('balance').
marketing_keyword('healthy').
marketing_keyword('slim').

critical_ratio(sugar_fat, 0.5, 2.0).

primary_function(dairy, protein_source).
primary_function(cereal, energy_source).
primary_function(beverage, hydration).
primary_function(snack, treat).
primary_function(processed_meat, protein_source).
primary_function(condiment, flavoring).

supports_function(protein_source, protein, Value) :- Value >= 10.
supports_function(energy_source, carbohydrate, Value) :- Value >= 40.
supports_function(hydration, sugar, Value) :- Value =< 5.