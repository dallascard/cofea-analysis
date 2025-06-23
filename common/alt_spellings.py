from re import M

# Choosing those that have an archaic form, or those that 
# have a similar replacement at 5%+ frequency

alt_spellings = {
    "ambassadors": ["embassadors"],
    "among": ["amongst"],
    "authorized": ["authorised"],
    "behavior": ["behaviour"],
    "cannot": ["canot"],
    "choose": ["chuse"],
    "choosing": ["chusing"],
    "compel": ["compell"],
    "control": ["controul"],
    "controversy": ["controversie"],
    "days": ["dayes"],
    "defense": ["defence"],
    "domestic": ["domestick"],
    "increase": ["encrease"],
    "entered": ["entred"],
    "expel": ["expell"],
    "favor": ["favour"],
    "guarantee": ["guaranty"],
    "habeas": ["habeus"],
    "honor": ["honour"],
    "inferior": ["inferiour"],
    "judgment": ["judgement"],
    "labor": ["labour"],
    "limited": ["limitted"],
    "massachusetts": ["massachusets"],
    "misdemeanors": ["misdemeanours"],
    "needful": ["needfull"],
    "net": ["nett"],
    "offense": ["offence"],
    "offenses": ["offences"],
    "organizing": ["organising"],
    "payment": ["paiment"],
    "pennsylvania": ["pensylvania"],
    "piracies": ["pyracies"],
    "privilege": ["priviledge"],
    "privileged": ["priviledged"],
    "privileges": ["priviledges"],
    "public": ["publick"],
    "receive": ["recieve"],
    "repel": ["repell"],
    "rhode": ["rhoad"],
    "secrecy": ["secresy"],
    "soldier": ["souldier"],
    "supreme": ["supream"],
    "swear": ["sware"],
    "tranquility": ["tranquillity"],
    "trial": ["tryal"],
    "tried": ["tryed"],
    "until": ["untill"],
    "useful": ["usefull"],
    "vessels": ["vessells"],
    "welfare": ["wellfare"],
    "writs": ["writts"],
}

alt_spellings_tokenized = {
  "ambassadors": [
    "em##bas##sa##dor##s"
  ],
  "among": [
    "amongst"
  ],
  "authorized": [
    "authorised"
  ],
  "behavior": [
    "behaviour"
  ],
  "cannot": [
    "can##ot"
  ],
  "choose": [
    "chu##se"
  ],
  "choosing": [
    "chu##sing"
  ],
  "com##pel": [
    "com##pel##l"
  ],
  "control": [
    "con##tro##ul"
  ],
  "controversy": [
    "con##tro##vers##ie"
  ],
  "days": [
    "day##es"
  ],
  "defense": [
    "defence"
  ],
  "domestic": [
    "domestic##k"
  ],
  "increase": [
    "en##cre##ase"
  ],
  "entered": [
    "en##tre##d"
  ],
  "ex##pel": [
    "ex##pel##l"
  ],
  "favor": [
    "favour"
  ],
  "guarantee": [
    "gu##aran##ty"
  ],
  "ha##be##as": [
    "ha##be##us"
  ],
  "honor": [
    "honour"
  ],
  "inferior": [
    "in##fer##io##ur"
  ],
  "judgment": [
    "judgement"
  ],
  "labor": [
    "labour"
  ],
  "limited": [
    "limit##ted"
  ],
  "massachusetts": [
    "mass##ach##use##ts"
  ],
  "mis##de##me##anor##s": [
    "mis##de##me##ano##urs"
  ],
  "need##ful": [
    "need##ful##l"
  ],
  "net": [
    "net##t"
  ],
  "offense": [
    "offence"
  ],
  "offenses": [
    "offences"
  ],
  "organizing": [
    "organising"
  ],
  "payment": [
    "pa##ime##nt"
  ],
  "pennsylvania": [
    "pens##yl##vani##a"
  ],
  "pi##rac##ies": [
    "p##yra##cies"
  ],
  "privilege": [
    "pri##vil##edge"
  ],
  "privileged": [
    "pri##vil##edge##d"
  ],
  "privileges": [
    "pri##vil##edge##s"
  ],
  "public": [
    "public##k"
  ],
  "receive": [
    "rec##ie##ve"
  ],
  "rep##el": [
    "rep##ell"
  ],
  "rhode": [
    "r##ho##ad"
  ],
  "secrecy": [
    "sec##res##y"
  ],
  "soldier": [
    "soul##dier"
  ],
  "supreme": [
    "su##pre##am"
  ],
  "swear": [
    "sw##are"
  ],
  "tran##quil##ity": [
    "tran##quil##lity"
  ],
  "trial": [
    "try##al"
  ],
  "tried": [
    "try##ed"
  ],
  "until": [
    "until##l"
  ],
  "useful": [
    "useful##l"
  ],
  "vessels": [
    "vessel##ls"
  ],
  "welfare": [
    "well##fare"
  ],
  "writ##s": [
    "writ##ts"
  ]
}


def get_replacements(tokenized=False):
    replacements = {}
    if tokenized:
        for primary, others in alt_spellings_tokenized.items():
            for other in others:
                replacements[other] = primary    
    else:
        for primary, others in alt_spellings.items():
            for other in others:
                replacements[other] = primary    
    return replacements