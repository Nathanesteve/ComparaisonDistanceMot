import main as mn
import numpy as np
mn.rng.seed(123456)


def tu_distance_jaro_winkler():

    test_1, test_2, test_3, test_4 = False, False, False, False
    a, b = mn.distance_jaro_winkler('dormir', 'odrmis', p=0.5)
    a = round(a, 3)
    b = round(b, 3)

    if (a == 0.822 and b == 0.822):
        test_1 = True
    else:
        print('ERREUR fonction distance_jaro_winkler test 1 ERREUR')
    a, b = mn.distance_jaro_winkler('MARTHA', 'MARHTA', p=0.1)
    a = round(a, 3)
    b = round(b, 3)
    if (a == 0.944 and b == 0.961):
        test_2 = True
    else:
        print('ERREUR fonction distance_jaro_winkler test 2 ERREUR')
    a, b = mn.distance_jaro_winkler('DWAYNE', 'DUANE', p=0.1)
    a = round(a, 3)
    b = round(b, 3)
    if (a == 0.822 and b == 0.840):
        test_3 = True
    else:
        print('ERREUR fonction distance_jaro_winkler test 3 ERREUR')
    a, b = mn.distance_jaro_winkler('DIXON', 'DICKSONX', p=0.1)
    a = round(a, 3)
    b = round(b, 3)
    if (a == 0.767 and b == 0.813):
        test_4 = True
    else:
        print('ERREUR fonction distance_jaro_winkler test 4 ERREUR')

    return(test_1 and test_2 and test_3 and test_4)


def tu_distance_levenshtein():

    test_1, test_2 = False, False
    a = mn.distance_levenshtein('chiens', 'niche')
    if (a == 5):
        test_1 = True
    else:
        print('ERREUR fonction distance_levenshtein test 1 ERREUR')
    b = mn.distance_levenshtein('croire', 'croître')
    if (b == 2):
        test_2 = True
    else:
        print('ERREUR fonction distance_levenshtein test 2 ERREUR')

    return(test_1 and test_2)


def tu_alteration_accent():

    test_1, test_2, test_3 = False, False, False
    test_4, test_5, test_6 = False, False, False

    if(mn.alteration_accent('aâà') == 'aaa'):
        test_1 = True
    else:
        print('ERREUR fonction alteration_accent test 1 ERREUR')

    if(mn.alteration_accent('uûù') == 'uuu'):
        test_2 = True
    else:
        print('ERREUR fonction alteration_accent test 2 ERREUR')

    if(mn.alteration_accent('eéèê') == 'eeee'):
        test_3 = True
    else:
        print('ERREUR fonction alteration_accent test 3 ERREUR')

    if(mn.alteration_accent('iîï') == 'iii'):
        test_4 = True
    else:
        print('ERREUR fonction alteration_accent test 4 ERREUR')

    if(mn.alteration_accent('oô') == 'oo'):
        test_5 = True
    else:
        print('ERREUR fonction alteration_accent test 5 ERREUR')

    if(mn.alteration_accent('abcdefghijklmnopqrstuvwxyz')
       == 'abcdefghijklmnopqrstuvwxyz'):
        test_6 = True
    else:
        print('ERREUR fonction alteration_accent test 6 ERREUR')

    return(test_1 and test_2 and test_3 and test_4 and test_5 and test_6)


def Test_Unitaire_Non_Regression():
    print(tu_distance_levenshtein())
    print(tu_distance_jaro_winkler())
    print(tu_alteration_accent())


print(tu_distance_levenshtein())
print(tu_distance_jaro_winkler())
print(tu_alteration_accent())

