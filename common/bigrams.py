
# (N*, N*) or (J*, N*) with NPMI >= 0.6 and at least 2000 mentions in COFEA
# exclude a few that are part of longer phrases
# Including titles, but not person names
# Excluding place names below the level of states
# Excluding greetings and numbers

bigrams = {
    ('nova', 'scotia'),    # NN  0.9918063516468822  3010
    ('united', 'states'),    # NN  0.9250710013851797  145086
    ('rhode', 'island'),    # NN  0.8836058710575944  14042
    ('new', 'york'),    # NN  0.8218812748616265  66155
    ('south', 'carolina'),    # NN  0.8212886933712444  15966
    ('west', 'indies'),    # NN  0.818999909186976   8809
    ('head', 'quarters'),    # NN  0.7754784720509029  12090
    ('reasonable', 'charges'),    # JN  0.7707669641004743  8750
    ('u.', 's.'),    # NN  0.7613493157081046  3466
    ('north', 'carolina'),    # NN  0.7600316882501063  12787
    ('yellow', 'fever'),    # JN  0.7549001941725396  2291
    ('small', 'pox'),    # JN  0.7532544683435047  4602
    ('minister', 'plenipotentiary'),    # NN  0.7466760995434504  2930
    ('fellow', 'citizens'),    # JN  0.728637777461306   8278
    ('great', 'britain'),    # NN  0.7169708603595453  35462
    ('common', 'pleas'),    # NN  0.7109335305409854  4388
    ('west', 'india'),    # NN  0.6968241538527291  3965
    ('new', 'jersey'),    # NN  0.6966630758225938  16867
    ('new', 'hampshire'),    # NN  0.6752512123544724  10289
    ('continental', 'congress'),    # NN  0.673319019830833   12059
    ('east', 'india'),    # NN  0.6677660832311952  2446
    ('lieutenant', 'colonel'),    # NN  0.6595595674303254  2819
    ('court', 'martial'),    # NN  0.6562709981102613  5150
    ('grand', 'jury'),    # NN  0.6454868240271631  2101
    ('military', 'stores'),    # JN  0.6442629521613247  2656
    ('indian', 'corn'),    # NN  0.6377603199340666  2149
    ('dwelling', 'house'),    # NN  0.6320804751008419  6707
    ('human', 'race'),    # JN  0.6303447899006736  2006
    ('fellow', 'creatures'),    # NN  0.6249598868138154  2088
    ('pounds', 'sterling'),    # NN  0.6185696682107317  2236
    ('quarter', 'master'),    # NN  0.6138629964677025  4532
    ('holy', 'scriptures'),    # NN  0.6050256565215881  2669
    ('foreign', 'affairs'),    # NN  0.6048609857990962  4620
    ('french', 'republic'),    # NN  0.6044713436375047  3219
    ('vice', 'president'),    # NN  0.60387767030691    3998
    ('public', 'use'),  # from DDO
    ('natural', 'born'),  # from DDO
    ('domestic', 'violence')  # from DDO
}

bigrams_tokenized = {
    ('united', 'states'),
    ('french', 'republic'),
    ('domestic', 'violence'),
    ('natural', 'born'),
    ('indian', 'corn'),
    ('lieutenant', 'colonel'),
    ('fellow', 'creatures'),
    ('south', 'carolina'),
    ('rhode', 'island'),
    ('u.', 's.'),
    ('foreign', 'affairs'),
    ('dwelling', 'house'),
    ('east', 'india'),
    ('west', 'indies'),
    ('minister', 'pl##eni##pot##ent##iary'),
    ('nova', 'scotia'),
    ('pounds', 'sterling'),
    ('fellow', 'citizens'),
    ('court', 'martial'),
    ('continental', 'congress'),
    ('yellow', 'fever'),
    ('small', 'po##x'),
    ('human', 'race'),
    ('new', 'hampshire'),
    ('west', 'india'),
    ('head', 'quarters'),
    ('reasonable', 'charges'),
    ('vice', 'president'),
    ('quarter', 'master'),
    ('holy', 'scriptures'),
    ('new', 'york'),
    ('common', 'pleas'),
    ('military', 'stores'),
    ('new', 'jersey'),
    ('grand', 'jury'),
    ('great', 'britain'),
    ('north', 'carolina'),
    ('public', 'use')
}
