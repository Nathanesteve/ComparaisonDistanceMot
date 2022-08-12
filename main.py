import random as rng
import numpy as np
import pandas as pd

# Variables

ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

# Fonctions alteration de mot


def alteration_orthographique(mot, p):

    '''

    Fonction qui prend en argument un mot et un parametre
    d'altération p pour retourner le mot modifié orthographiquement

    '''

    taille_mot = len(mot)
    mot_alteration_orthographique = ''
    lettre = ''

    for i in range(0, taille_mot):
        lettre = mot[i]
        proba = rng.uniform(0, 1)
        if proba < p:
            lettre = rng.choice(ALPHABET)
        mot_alteration_orthographique = mot_alteration_orthographique + lettre

    return mot_alteration_orthographique


def alteration_troncature(mot, t):

    '''

    Fonction qui prend en argument un mot et un parametre
    de troncature t pour retourner le mot tronqué

    '''

    taille_mot = len(mot)
    nb_troncature = round(taille_mot * t)

    if nb_troncature == 0:
        nb_troncature = 1
    mot_alteration_troncature = mot[0:-nb_troncature]

    return mot_alteration_troncature


def alteration_accent(mot):

    '''

    Fonction qui prend en argument un mot et qui
    retourne le mot sans accent

    '''

    taille_mot = len(mot)
    mot_alteration_accent = ''
    lettre = ''

    for i in range(0, taille_mot):
        lettre = mot[i]
        if lettre in ['â', 'à']:
            lettre = 'a'
        if lettre in ['û', 'ù']:
            lettre = 'u'
        if lettre in ['é', 'è', 'ê']:
            lettre = 'e'
        if lettre in ['î', 'ï']:
            lettre = 'i'
        if lettre in ['ô']:
            lettre = 'o'
        mot_alteration_accent = mot_alteration_accent + lettre

    return mot_alteration_accent


def alteration_data(data, type, param_alteration):

    '''

    Fonction qui prend en argument un jeu de donnée, un type d'altération
    et un parametre d'altération et qui retourne le jeu données ou chaque
    mot asubit modification choisit.

    '''

    n_data = []

    if type == 'accent':
        for i in data:
            n_mot = 0
            n_mot = alteration_accent(i)
            n_data.append(n_mot)

    if type == 'orthographe':
        for i in data:
            n_mot = 0
            n_mot = alteration_orthographique(i, param_alteration)
            n_data.append(n_mot)

    if type == 'troncature':
        for i in data:
            n_mot = 0
            n_mot = alteration_troncature(i, param_alteration)
            n_data.append(n_mot)

    np.array(n_data)
    alter_data = np.column_stack((n_data, data))
    return alter_data


# Distances


def correpondance_jaro_winkler(mot1, mot2, lettre, position):

    '''

    Fonction qui determine si une lettre est correpondante

    '''

    taille1 = len(mot1)
    taille2 = len(mot2)

    c = round((max(abs(taille1), abs(taille2))/2) - 1)

    selection = mot2[max(0, position - c): c+position]

    liste_lettre_correpondantes = []
    for i in selection:
        liste_lettre_correpondantes.append(i)

    correspondance = (lettre in liste_lettre_correpondantes)

    return correspondance


def coefficient_mat_jaro_winkler(mot1, mot2):

    '''

    Donne le parametre c et la liste des lettres identiques 
    et correspondantes de deux mots

    '''

    if len(mot1) != len(mot2):
        jaro1 = max(mot1, mot2, key=len)
        jaro2 = min(mot1, mot2, key=len)
    elif len(mot1) == len(mot2):
        jaro1 = mot1
        jaro2 = mot2

    count_coef_jaro_winkler = 0
    lst_correspondance = []

    for i in range(0, len(jaro1)):
        if correpondance_jaro_winkler(jaro1, jaro2, jaro1[i], i):
            count_coef_jaro_winkler = count_coef_jaro_winkler + 1
            lst_correspondance.append(jaro1[i])

    return count_coef_jaro_winkler, lst_correspondance


def cacl_param_t_jaro_winkler(mot1, mot2, correspondance):

    '''

    Calcul du parametre t de Jaro-Winkler de deux mots

    '''

    selection1 = []
    selection2 = []

    for i in mot1:
        selection1.append(i)

    for j in mot2:
        selection2.append(j)

    selection1_inter = [i for i in selection1 if i in correspondance]
    selection2_inter = [i for i in selection2 if i in correspondance]

    lst = selection1 + selection2
    v2 = [n for n in lst if lst.count(n) == 1]

    # v2 correspond a la liste des lettres de mot1 absente dans mot2

    selection1_final = [i for i in selection1_inter if i not in v2]
    selection2_final = [i for i in selection2_inter if i not in v2]

    count = 0
    for i in range(0, min(len(selection1_final), len(selection2_final))):
        if selection1_final[i] != selection2_final[i]:
            count = count + 1

    t = count/2

    return(t)


def cacl_distance_jaro_winkler(mot1, mot2, m, t, p=0.1):

    '''

    Calcul de la distance de Jaro et Jaro-Winkler de deux mots
    à partir de deux mots et du parametre m et p

    '''

    taille1 = len(mot1)
    taille2 = len(mot2)
    param_l = 0
    count = 0

    min_len = min(4, taille1, taille2)
    for i in range(0, min_len):

        if (mot1[i] == mot2[i] and count == param_l):
            param_l = param_l + 1
        count = count + 1

    if m != 0:
        distance_jaro = ((1/3) * ((m/taille1) + (m/taille2) + ((m-t)/m)))
        distance_Jaro_Winkler = (distance_jaro + (param_l*p*(1-distance_jaro)))
    else:
        distance_jaro = 0
        distance_Jaro_Winkler = 0

    return distance_jaro, distance_Jaro_Winkler


def distance_jaro_winkler(mot1, mot2, p=0.1):

    '''

    Determine la distance de Jaro et Jaro-Winkler de deux mots

    '''

    m, lst = coefficient_mat_jaro_winkler(mot1, mot2)
    t = cacl_param_t_jaro_winkler(mot1, mot2, lst)
    dj, dw = cacl_distance_jaro_winkler(mot1, mot2, m, t, p)

    return dj, dw


def distance_levenshtein(chaine1, chaine2):

    '''

    Determine la Levenshtein de deux mots

    Source Algo : https://fr.wikipedia.org/wiki/Distance_de_Levenshtein

    '''

    M = len(chaine1) + 1
    N = len(chaine2) + 1
    matrice_levenshtein = np.zeros((M, N))

    for x in range(M):
        matrice_levenshtein[x, 0] = x
    for y in range(N):
        matrice_levenshtein[0, y] = y
    for x in range(1, M):
        for y in range(1, N):
            if chaine1[x-1] == chaine2[y-1]:
                matrice_levenshtein[x, y] = min(
                    matrice_levenshtein[x-1, y] + 1,
                    matrice_levenshtein[x-1, y-1],
                    matrice_levenshtein[x, y-1] + 1)
            else:
                matrice_levenshtein[x, y] = min(
                    matrice_levenshtein[x-1, y] + 1,
                    matrice_levenshtein[x-1, y-1] + 1,
                    matrice_levenshtein[x, y-1] + 1)

    return (matrice_levenshtein[M-1, N-1])


def prediction_mot_jaro_winkler(mot1, data_set, p):

    '''

    Determine le mot le plus proche de mot1 selon Jaro et Jaro-Winkler
    à partir d'un data_set


    '''

    liste_distance_dj = []
    liste_distance_dw = []

    for i in data_set:
        mot2 = i
        dj, dw = distance_jaro_winkler(mot1, mot2, p)
        liste_distance_dj.append(dj)
        liste_distance_dw.append(dw)

    array_distance_dj = np.array(liste_distance_dj)
    array_distance_dw = np.array(liste_distance_dw)

    n = np.argmax(array_distance_dj)
    m = np.argmax(array_distance_dw)

    mot_le_plus_proche_dj = data_set[n]
    mot_le_plus_proche_dw = data_set[m]

    return mot_le_plus_proche_dj, mot_le_plus_proche_dw


def prediction_vecteur_jaro_winkler(mat, data_set, p):

    '''

    Determine le vecteur des mots les plus proche de mat
    selon Jaro et Jaro-Winkler à partir d'un data_set

    '''

    list_mot_le_plus_proche_dj = []
    list_mot_le_plus_proche_dw = []

    for i in mat.T[0]:
        a, b = prediction_mot_jaro_winkler(i, data_set, p)
        list_mot_le_plus_proche_dj.append(''.join(a))
        list_mot_le_plus_proche_dw.append(''.join(b))

    return list_mot_le_plus_proche_dj, list_mot_le_plus_proche_dw


def calc_bonne_prediction(matrice_mots, distance):

    '''

    Cacule le score de bonne prédiction obtenue avec la distance choisit

    '''

    if distance == 'dj':
        PARAM = 2
    if distance == 'dw':
        PARAM = 3
    if distance == 'dlev':
        PARAM = 2

    pred = []
    nb_ligne = np.size(matrice_mots, axis=0)

    for i in range(0, nb_ligne):
        if matrice_mots[i, 1] == matrice_mots[i, PARAM]:
            pred.append(1)
        else:
            pred.append(0)

    prediction = (sum(pred)/len(pred))
    return prediction


def calc_bonne_prediction_total(matrice_mots, distance):

    if distance == 'dj':
        PARAM = 2
    if distance == 'dw':
        PARAM = 3
    if distance == 'dlev':
        PARAM = 4
    

    pred = []
    nb_ligne = np.size(matrice_mots, axis=0)

    for i in range(0, nb_ligne):
        if matrice_mots[i, 1] == matrice_mots[i, PARAM]:
            pred.append(1)
        else:
            pred.append(0)

    prediction = (sum(pred)/len(pred))
    return prediction


def resultat_jaro_winkler(data_set_altere, data_set,
                          show_info=True, p=0.1):

    array_dj, array_dw = prediction_vecteur_jaro_winkler(data_set_altere,
                                                         data_set,
                                                         p)

    matrice_mots = np.column_stack((data_set_altere, array_dj, array_dw))

    resultat_dj = calc_bonne_prediction(matrice_mots, 'dj')
    resultat_dw = calc_bonne_prediction(matrice_mots, 'dw')

    if show_info:
        df = pd.DataFrame(matrice_mots,
                          columns=['Mot altéré    ',
                                   '    Mot    ',
                                   '    Correspondance Jaro    ',
                                   '    Correspondance Jaro-Winkler    '])
        print(df)
        print(f'Jaro bonne prediction: {resultat_dj * 100}%')
        print(f'Jaro-Winkler bonne prediction: {resultat_dw * 100}%')

    return resultat_dj, resultat_dw, matrice_mots


def prediction_mot_levenshtein(mot1, data_set):

    liste_distance_lev = []

    for i in data_set:
        mot2 = i
        dlev = distance_levenshtein(mot1, mot2)
        liste_distance_lev.append(dlev)

    array_distance_lev = np.array(liste_distance_lev)

    n = np.argmin(array_distance_lev)
    mot_le_plus_proche_dlev = data_set[n]
    return mot_le_plus_proche_dlev


def prediction_vecteur_levenshtein(mat, data_set):
    '''

    Produit le vecteur contenant le mot le plus proche 

    '''

    list_mot_le_plus_proche_dlev = []

    for i in mat.T[0]:
        a = prediction_mot_levenshtein(i, data_set)
        list_mot_le_plus_proche_dlev.append(''.join(a))

    return list_mot_le_plus_proche_dlev


def resultat_levenshtein(data_set_altere, data_set, show_info=True):

    array_dlev = prediction_vecteur_levenshtein(data_set_altere, data_set)
    matrice_mots = np.column_stack((data_set_altere, array_dlev))
    result = calc_bonne_prediction(matrice_mots, 'dj')

    if show_info:
        df = pd.DataFrame(matrice_mots,
                          columns=['Mot altéré    ',
                                   '    Mot    ',
                                   '    Correspondance Levenshtein    '])
        print(df)
        print(f'Levenshtein bonne prediction {result * 100}%')

    return result, matrice_mots


def resultat_total(data_set_altere, data_set,
                   show_info=False, p=0.1):

    array_dj, array_dw = prediction_vecteur_jaro_winkler(data_set_altere,
                                                         data_set,
                                                         p)
    array_dlev = prediction_vecteur_levenshtein(data_set_altere, data_set)

    matrice_mots = np.column_stack((data_set_altere,
                                    array_dj,
                                    array_dw,
                                    array_dlev))

    resultat_dj = calc_bonne_prediction_total(matrice_mots, 'dj')
    resultat_dw = calc_bonne_prediction_total(matrice_mots, 'dw')
    resultat_dlev = calc_bonne_prediction_total(matrice_mots, 'dlev')

    if show_info:
        df = pd.DataFrame(matrice_mots,
                          columns=['Mot altéré    ',
                                   '    Mot    ',
                                   '    Correspondance Jaro    ',
                                   '    Correspondance Jaro-Winkler    ',
                                   '    Correspondance Levenshtein    '])
        print(df)
        print(f'Jaro bonne prediction: {resultat_dj * 100}%')
        print(f'Jaro-Winkler bonne prediction: {resultat_dw * 100}%')
        print(f'Levenshtein bonne prediction: {resultat_dlev * 100}%')

    return resultat_dj, resultat_dw, resultat_dlev, matrice_mots


# Fonction de resultat

def score_prediction_levenshtein(data, type_alteration,
                                 param_alteration, nb_iter, show_info=True):

    '''

    Cette fonction prends en entrée un jeu de données, un type d'altération,
    un parametre d'altération et un nombre d'itéraiton n puis construit n jeux
    de données altéré pour calculer le score de bonne prédiction de la distance
    de Levenshtein

    '''

    sto_result = []
    for i in range(0, nb_iter):
        data_altere = alteration_data(data, type_alteration, param_alteration)
        levenshtein, resultat = resultat_levenshtein(data_altere,
                                                     data, show_info)
        sto_result.append(levenshtein)
    return (sum(sto_result)/nb_iter)


def score_prediction_jaro_winkler(data, type_alteration,
                                  param_alteration,
                                  nb_iter, p=0.1,
                                  show_info=True):

    '''

    Cette fonction prends en entrée un jeu de données, un type d'altération,
    un parametre d'altération et un nombre d'itéraiton n puis construit n jeux
    de données altéré pour calculer le score de bonne prédiction de la distance
    de Jaro et Jaro-Winkler

    '''

    sto_result_jaro = []
    sto_result_jaro_winkler = []

    for i in range(0, nb_iter):

        data_altere = alteration_data(data,
                                      type_alteration,
                                      param_alteration)

        jaro, jaro_winkler, resultat = resultat_jaro_winkler(data_altere,
                                                             data,
                                                             show_info,
                                                             p)

        sto_result_jaro.append(jaro)
        sto_result_jaro_winkler.append(jaro_winkler)

    prediction_jaro = (sum(sto_result_jaro)/nb_iter)
    prediction_jaro_winkler = (sum(sto_result_jaro_winkler)/nb_iter)
    return prediction_jaro, prediction_jaro_winkler


def score_prediction_total(data, type_alteration,
                           param_alteration,
                           nb_iter, p=0.1,
                           show_info=False):

    '''

    Cette fonction prends en entrée un jeu de données, un type d'altération,
    un parametre d'altération et un nombre d'itéraiton n puis construit n jeux
    de données altéré pour calculer le score de bonne prédiction de la distance
    de Jaro, Jaro-Winkler et Levenshtein.

    '''

    sto_result_jaro = []
    sto_result_jaro_winkler = []
    sto_result_levenshtein = []

    for i in range(0, nb_iter):

        data_altere = alteration_data(data,
                                      type_alteration,
                                      param_alteration)

        jaro, jaro_winkler, levenshtein, resultat = resultat_total(data_altere,
                                                             data,
                                                             show_info,
                                                             p)

        sto_result_jaro.append(jaro)
        sto_result_jaro_winkler.append(jaro_winkler)
        sto_result_levenshtein.append(levenshtein)

    prediction_jaro = (sum(sto_result_jaro)/nb_iter)
    prediction_jaro_winkler = (sum(sto_result_jaro_winkler)/nb_iter)
    prediction_levenshtein = (sum(sto_result_levenshtein)/nb_iter)
    return prediction_jaro, prediction_jaro_winkler, prediction_levenshtein


























def score_prediction_jaro_winkler_test(data, p=0.1,
                                       show_info=True):

    sto_result_jaro = []
    sto_result_jaro_winkler = []

    for i in range(0, 5):

        data_altere_1 = np.array(['odrmis', 'MARHTA',
                                  'DUANE', 'DICKSONX',
                                  'niche', 'croître',
                                  'darmir', 'dirmir'])

        data_altere_2 = np.column_stack((data_altere_1, data))

        jaro, jaro_winkler, resultat = resultat_jaro_winkler(data_altere_2,
                                                             data,
                                                             show_info,
                                                             p)

        sto_result_jaro.append(jaro)
        sto_result_jaro_winkler.append(jaro)

    prediction_jaro = (sum(sto_result_jaro)/5)
    prediction_jaro_winkler = (sum(sto_result_jaro_winkler)/5)
    return prediction_jaro, prediction_jaro_winkler
