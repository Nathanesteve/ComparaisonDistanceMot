import main as mn
import time
import random as rng
import numpy as np


mn.rng.seed(123456)

print(mn.alteration_orthographique('aléatoire', p=0))
print(mn.alteration_orthographique('aléatoire', p=0.1))
print(mn.alteration_orthographique('aléatoire', p=0.2))
print(mn.alteration_orthographique('aléatoire', p=0.3))
print(mn.alteration_orthographique('aléatoire', p=0.4))
print(mn.alteration_orthographique('aléatoire', p=0.5))


print(mn.alteration_troncature('troncature', t=0.1))
print(mn.alteration_troncature('troncature', t=0.2))
print(mn.alteration_troncature('troncature', t=0.4))
print(mn.alteration_troncature('troncature', t=0.5))
print(mn.alteration_troncature('troncature', t=0.6))


print("begin")
f = open('data_set_complet.txt', encoding='utf_8')
content = f.read()
content_list = content.split()

np.random.seed(123456)
rng.seed(123456)


type_alteration = 'orthographe'
param_alteration = 0.5
iter = 10
N = 10
mots = 100
time_start = time.time()
result_stack = np.zeros((1, 3))
P = 0.1
for i in range(0, N):
    print(f'PROCESS.............{i}/{N}')
    data = np.random.choice(content_list, mots, False)
    result = mn.score_prediction_total(data, type_alteration, param_alteration,
                                       iter, p=P, show_info=False)
    result_stack = np.vstack((result_stack, np.array(result)))
    print(result)
time_stop = time.time()

print(time_start)
print(time_stop)
print('=================================================')
print('Parametrage:')
print(f'Taille echantillon = {mots}')
print(f'type_alteration = {type_alteration}')
print(f'param_alteration = {param_alteration}')
print(f'nb_iter = {iter}')
print(f'N = {N}')
print(f'p = {P}')


s = np.sum(result_stack, axis=0)
print('Jaro, Jaro-Winkler, levenshtein')
print(s/N)

print(f"Temps d'exécution : {time_stop - time_start}s ")
print('=================================================')

