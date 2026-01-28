:- ensure_loaded('facts.pl').

% REGOLE BASE

is_empty_calories(Sugar, Fiber) :-
    threshold(sugar, high, T_Sugar),
    threshold(fiber, good, T_Fiber),
    Sugar > T_Sugar,
    Fiber < T_Fiber.

is_hidden_sodium(Salt, FruitVeg) :-
    threshold(salt, high, T_Salt),
    threshold(fruit_veg, low, T_Fruit),
    Salt > T_Salt,
    FruitVeg < T_Fruit.

is_hyper_processed(Additives, _, _) :-
    threshold(additives, max, T_Add),
    Additives > T_Add.
is_hyper_processed(_, Sugar, Salt) :-
    threshold(sugar, very_high, T_Sugar),
    threshold(salt, high, T_Salt),
    Sugar > T_Sugar,
    Salt > T_Salt.

is_low_fat_sugar_trap(Fat, Sugar) :-
    threshold(fat, low, T_Fat),
    threshold(sugar, high, T_Sugar),
    Fat < T_Fat,
    Sugar > T_Sugar.

is_misleading_label(NameAtom, Sugar, Salt) :-
    threshold(sugar, high, T_Sugar),
    threshold(salt, high, T_Salt),
    (Sugar > T_Sugar ; Salt > T_Salt),
    downcase_atom(NameAtom, NameLower),
    marketing_keyword(Key),
    sub_atom(NameLower, _, _, _, Key),
    !.

is_hyperpalatable(Sugar, Fat) :-
    Sugar > 0, Fat > 0,
    Ratio is Sugar / Fat,
    critical_ratio(sugar_fat, MinR, MaxR),
    Ratio >= MinR, Ratio =< MaxR,
    threshold(sugar, high, T_S), threshold(fat, high, T_F),
    Sugar > T_S, Fat > T_F.

is_protein_wash(NameAtom, Protein) :-
    downcase_atom(NameAtom, NameLower),
    (sub_atom(NameLower, _, _, _, 'protein') ; sub_atom(NameLower, _, _, _, 'sport')),
    threshold(protein, high, T_Prot),
    Protein < T_Prot.

is_ultra_processed_combo(Additives, Sugar, Salt) :-
    threshold(additives, max, T_A), threshold(sugar, high, T_S), threshold(salt, high, T_Na),
    Additives > T_A, Sugar > T_S, Salt > T_Na.


infer_category_from_name(NameAtom, Category) :-
    downcase_atom(NameAtom, NameLower),
    keyword_category(Keyword, Category),
    sub_atom(NameLower, _, _, _, Keyword),
    !.
infer_category_from_name(_, unknown).

violates_category_expectation(NameAtom, Nutrient, Value) :-
    infer_category_from_name(NameAtom, Category),
    Category \= unknown,
    inherits_expectation(Category, Nutrient, Min, Max),
    (Value > Max ; Value < Min).


meets_function(energy_source, _, Sugar, Carbs) :-
    supports_function(energy_source, carbohydrate, MinCarbs),
    Carbs >= MinCarbs,
    Sugar < 15.

meets_function(protein_source, Protein, _, _) :-
    supports_function(protein_source, protein, MinProt),
    Protein >= MinProt.

meets_function(hydration, _, Sugar, _) :-
    supports_function(hydration, sugar, MaxSugar),
    Sugar =< MaxSugar.

meets_function(treat, _, _, _).
meets_function(flavoring, _, _, _).
meets_function(condiment, _, _, _).


is_functionally_inconsistent(NameAtom, Protein, Sugar, Carbs) :-
    infer_category_from_name(NameAtom, Category),
    Category \= unknown,
    is_a(Category, ParentCategory),
    primary_function(ParentCategory, Function),
    \+ meets_function(Function, Protein, Sugar, Carbs).

% PESI (Weights)

weight(is_false_health_claim, 6).
weight(is_misleading_label, 5).
weight(is_ultra_processed_combo, 5).
weight(is_functionally_inconsistent, 5).
weight(is_hyper_processed, 4).
weight(is_hyperpalatable, 4).
weight(is_empty_calories, 4).
weight(violates_category_expectation, 3).
weight(is_protein_wash, 3).
weight(is_hidden_sodium, 3).
weight(is_low_fat_sugar_trap, 3).

% CALCOLO SCORE

compute_risk_score(Sugar, Fat, Salt, Fiber, FruitVeg, Additives, Protein, Carbs, Name, TotalScore) :-
    findall(W, (
        (is_empty_calories(Sugar, Fiber), weight(is_empty_calories, W));
        (is_hidden_sodium(Salt, FruitVeg), weight(is_hidden_sodium, W));
        (is_hyper_processed(Additives, Sugar, Salt), weight(is_hyper_processed, W));
        (is_low_fat_sugar_trap(Fat, Sugar), weight(is_low_fat_sugar_trap, W));
        (is_misleading_label(Name, Sugar, Salt), weight(is_misleading_label, W));
        (is_hyperpalatable(Sugar, Fat), weight(is_hyperpalatable, W));
        (violates_category_expectation(Name, sugar, Sugar), weight(violates_category_expectation, W));
        (violates_category_expectation(Name, protein, Protein), weight(violates_category_expectation, W));
        (violates_category_expectation(Name, salt, Salt), weight(violates_category_expectation, W));
        (is_functionally_inconsistent(Name, Protein, Sugar, Carbs), weight(is_functionally_inconsistent, W));
        (is_protein_wash(Name, Protein), weight(is_protein_wash, W));
        (is_ultra_processed_combo(Additives, Sugar, Salt), weight(is_ultra_processed_combo, W))
    ), Weights),
    sum_list(Weights, TotalScore).