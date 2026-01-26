:- ensure_loaded('facts.pl').

% --- DEFINIZIONE KEYWORDS (DATI) ---
marketing_keyword('bio').
marketing_keyword('vegan').
marketing_keyword('light').
marketing_keyword('fit').
marketing_keyword('natural').
marketing_keyword('detox').
marketing_keyword('sport').
marketing_keyword('gluten free').

% --- REGOLE DI BASE ---

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

is_high_satiety(Protein, Fiber) :-
    threshold(protein, good, T_Prot),
    threshold(fiber, good, T_Fiber),
    Protein > T_Prot,
    Fiber < T_Fiber.

is_low_fat_sugar_trap(Fat, Sugar) :-
    threshold(fat, low, T_Fat),
    threshold(sugar, high, T_Sugar),
    Fat < T_Fat,
    Sugar > T_Sugar.

% --- REGOLA AVANZATA: MARKETING INGANNEVOLE ---
is_misleading_label(NameAtom, Sugar, Salt) :-
    threshold(sugar, high, T_Sugar),
    threshold(salt, high, T_Salt),
    (Sugar > T_Sugar ; Salt > T_Salt),
    downcase_atom(NameAtom, NameLower),
    marketing_keyword(Key),     % Prende una keyword dalla lista
    sub_atom(NameLower, _, _, _, Key), % Cerca la keyword nel nome
    !. % Cut: Appena ne trova una, si ferma (evita duplicati)

% --- SYMBOLIC SCORE ---
% Conta quante regole negative sono attive per un prodotto.
% Restituisce un numero da 0 a N.
count_flags(Sugar, Fat, Salt, Fiber, FruitVeg, Additives, _, Name, Count) :-
    findall(1, (
        is_empty_calories(Sugar, Fiber);
        is_hidden_sodium(Salt, FruitVeg);
        is_hyper_processed(Additives, Sugar, Salt);
        is_low_fat_sugar_trap(Fat, Sugar);
        is_misleading_label(Name, Sugar, Salt)
    ), List),
    length(List, Count).