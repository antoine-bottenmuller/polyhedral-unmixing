#define PY_SSIZE_T_CLEAN
#include <Python.h> // Pour lier le code aux extensions Python

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h> // Pour manipuler les tableaux NumPy

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <math.h> // Inclure math.h: -lm
#include <glpk.h> // Inclure GLPK (avoir installé GLPK dans l'environnement: sudo apt-get install glpk-utils libglpk-dev glpk-doc): -lglpk
//#include <cblas.h> // Inclure la bibliothèque BLAS (avoir installé BLAS dans l'environnement: sudo apt-get install libopenblas-dev): -lopenblas

#include <time.h>


//////////////////////////////////////////////////// BASIC FUNCTIONS /////////////////////////////////////////////////////
// #######################################################################################################################
// #######################################################################################################################
// #######################################################################################################################

////////////////////////////////////////////// FUNCTIONS FOR LINEAR ALGEBRA //////////////////////////////////////////////
// #######################################################################################################################

// Fonction pour calculer le produit scalaire de deux vecteurs
double scalar(const double* a, const double* b, int size) {
    double result = 0.0;
    for (int j = 0; j < size; j++) {
        result += a[j] * b[j];
    }
    return result;
}

// Fonction pour calculer la norme d'un vecteur
double norm(const double* v, int size) {
    return sqrt(scalar(v, v, size));
}

// Fonction pour calculer la norme d'un array de vecteurs
double* norm_V(const double* V, int rows, int cols) {
    
    // Allouer de la mémoire pour le résultat
    double* result = (double*)malloc(rows * sizeof(double));
    if (result == NULL) {
        fprintf(stderr, "Erreur d'allocation memoire\n");
        exit(EXIT_FAILURE);
    }

    // Calculer la norme pour chaque ligne
    for (int i = 0; i < rows; i++) {
        result[i] = norm(V + i * cols, cols);
    }

    return result;
}

////////////////////////////////////////////// FUNCTIONS TO COMPUTE <q,V>-s //////////////////////////////////////////////
// #######################################################################################################################

// Fonction pour calculer < q - C, V > sur tous les indices de C,V
double* scalar_qVs(const double* V, const double* s, const double* q, int rows, int cols) {
    
    // Allouer de la mémoire pour le résultat
    double* result = (double*)malloc(rows * sizeof(double));
    if (result == NULL) {
        fprintf(stderr, "Erreur d'allocation memoire\n");
        exit(EXIT_FAILURE);
    }

    // Calculer le produit scalaire pour chaque ligne
    for (int i = 0; i < rows; i++) {
        result[i] = scalar(q, V + i * cols, cols) - s[i];
    }

    return result;
}

// Fonction pour calculer < q - C, V > sur les indices Id de C,V
double* scalar_qVs_I(const double* V, const double* s, const double* q, const int* Id, int rows_Id, int cols) {
    
    // Allouer de la mémoire pour le résultat
    double* result = (double*)malloc(rows_Id * sizeof(double));
    if (result == NULL) {
        fprintf(stderr, "Erreur d'allocation memoire\n");
        exit(EXIT_FAILURE);
    }

    // Calculer le produit scalaire pour chaque ligne
    for (int i = 0; i < rows_Id; i++) {
        int j = Id[i];
        result[i] = scalar(q, V + j * cols, cols) - s[j];
    }

    return result;
}

// Fonction pour calculer < q - C, V > sur les indices I de C,V, avec I = Id[Ip] 
double* scalar_qVs_IIp(const double* V, const double* s, const double* q, const int* Id, const int* Ip, int rows_Ip, int cols) {
    
    // Allouer de la mémoire pour le résultat
    double* result = (double*)malloc(rows_Ip * sizeof(double));
    if (result == NULL) {
        fprintf(stderr, "Erreur d'allocation memoire\n");
        exit(EXIT_FAILURE);
    }

    // Calculer le produit scalaire pour chaque ligne
    for (int i = 0; i < rows_Ip; i++) {
        int j = Id[Ip[i]];
        result[i] = scalar(q, V + j * cols, cols) - s[j];
    }

    return result;
}

////////////////////////////////////////////// MAXIMUM AND ARGMAX FUNCTIONS //////////////////////////////////////////////
// #######################################################################################################################

// Fonction pour calculer le maximum d'un tableau 1D
double max(const double* array, int size) {
    if (size <= 0) {
        fprintf(stderr, "Erreur : Taille du tableau invalide\n");
        return -1.0 / 0.0; // Retourne -inf pour indiquer une erreur
    }

    double max_value = array[0]; // Initialiser avec le premier élément
    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
        }
    }
    return max_value;
}

// Fonction pour calculer l'argmax d'un tableau 1D
int argmax(const double* array, int size) {
    if (size <= 0) {
        fprintf(stderr, "Erreur : Taille du tableau invalide\n");
        return -1.0 / 0.0; // Retourne -inf pour indiquer une erreur
    }

    int arg = 0;
    double max_value = array[0]; // Initialiser avec le premier element
    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            arg = i;
            max_value = array[i];
        }
    }
    return arg;
}

// Fonction pour calculer l'argmax d'un tableau 1D sur les indices I
int argmax_I(const double* array, const int* Id, int size_Id) {
    if (size_Id <= 0) {
        fprintf(stderr, "Erreur : Taille du tableau invalide\n");
        return -1.0 / 0.0; // Retourne -inf pour indiquer une erreur
    }

    int arg = Id[0];
    double max_value = array[arg]; // Initialiser avec le premier élément
    for (int i = 1; i < size_Id; i++) {
        int j = Id[i];
        if (array[j] > max_value) {
            arg = j;
            max_value = array[j];
        }
    }
    return arg;
}

// Fonction pour calculer l'argmax d'un tableau 1D sur les indices I = Id[Ip]
int argmax_I_i(const double* array, const int* Id, int size_Id) {
    if (size_Id <= 0) {
        fprintf(stderr, "Erreur : Taille du tableau invalide\n");
        return -1.0 / 0.0; // Retourne -inf pour indiquer une erreur
    }

    int arg = 0;
    double max_value = array[Id[0]]; // Initialiser avec le premier élément
    for (int i = 1; i < size_Id; i++) {
        double value = array[Id[i]];
        if (value > max_value) {
            arg = i;
            max_value = value;
        }
    }
    return arg;
}

// Fonction pour calculer l'argmax d'un tableau 1D sur les indices I = Id[Ip]
int argmax_IIp(const double* array, const int* Id, const int* Ip, int size_Ip) {
    if (size_Ip <= 0) {
        fprintf(stderr, "Erreur : Taille du tableau invalide\n");
        return -1.0 / 0.0; // Retourne -inf pour indiquer une erreur
    }

    int arg = Id[Ip[0]];
    double max_value = array[arg]; // Initialiser avec le premier élément
    for (int i = 1; i < size_Ip; i++) {
        int j = Id[Ip[i]];
        if (array[j] > max_value) {
            arg = j;
            max_value = array[j];
        }
    }
    return arg;
}

// Fonction pour calculer l'argmax d'un tableau 1D sur les indices I = Id[Ip]
int argmax_IIp_i(const double* array, const int* Id, const int* Ip, int size_Ip) {
    if (size_Ip <= 0) {
        fprintf(stderr, "Erreur : Taille du tableau invalide\n");
        return -1.0 / 0.0; // Retourne -inf pour indiquer une erreur
    }

    int arg = 0;
    double max_value = array[Id[Ip[0]]]; // Initialiser avec le premier élément
    for (int i = 1; i < size_Ip; i++) {
        double value = array[Id[Ip[i]]];
        if (value > max_value) {
            arg = i;
            max_value = value;
        }
    }
    return arg;
}

////////////////////////////////////////////// FUNCTION TO (ARG-)SORT LIST ///////////////////////////////////////////////
// #######################################################################################################################

// Structure pour associer une valeur et son indice
typedef struct {
    double value;
    int index;
} IndexedValue;

// Fonction de comparaison pour trier en ordre décroissant
int compare_desc(const void* a, const void* b) {
    double diff = ((IndexedValue*)b)->value - ((IndexedValue*)a)->value;
    return (diff > 0) - (diff < 0); // Retourne 1 si a < b, -1 si a > b, 0 si égal
}

// Fonction pour obtenir les indices triés
int* argsort_desc(const double* array, int size) {
    
    // Etape 1 : Créer un tableau de structures IndexedValue
    IndexedValue* indexed_array = (IndexedValue*)malloc(size * sizeof(IndexedValue));
    if (indexed_array == NULL) {
        fprintf(stderr, "Erreur d'allocation memoire\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++) {
        indexed_array[i].value = array[i];
        indexed_array[i].index = i;
    }

    // Etape 2 : Trier le tableau selon les valeurs en ordre décroissant
    qsort(indexed_array, size, sizeof(IndexedValue), compare_desc);

    // Etape 3 : Extraire les indices triés
    int* sorted_indices = (int*)malloc(size * sizeof(int));
    if (sorted_indices == NULL) {
        fprintf(stderr, "Erreur d'allocation memoire\n");
        free(indexed_array);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++) {
        sorted_indices[i] = indexed_array[i].index;
    }

    // Libérer la mémoire intermédiaire
    free(indexed_array);

    return sorted_indices;
}

// Fonction pour obtenir les indices triés sur Id
int* argsort_desc_I(const double* array, const int* Id, int size_Id) {
    
    // Etape 1 : Créer un tableau de structures IndexedValue
    IndexedValue* indexed_array = (IndexedValue*)malloc(size_Id * sizeof(IndexedValue));
    if (indexed_array == NULL) {
        fprintf(stderr, "Erreur d'allocation memoire\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size_Id; i++) {
        indexed_array[i].value = array[Id[i]];
        indexed_array[i].index = Id[i];
    }

    // Etape 2 : Trier le tableau selon les valeurs en ordre décroissant
    qsort(indexed_array, size_Id, sizeof(IndexedValue), compare_desc);

    // Etape 3 : Extraire les indices triés
    int* sorted_indices = (int*)malloc(size_Id * sizeof(int));
    if (sorted_indices == NULL) {
        fprintf(stderr, "Erreur d'allocation memoire\n");
        free(indexed_array);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size_Id; i++) {
        sorted_indices[i] = indexed_array[i].index;
    }

    // Libérer la mémoire intermédiaire
    free(indexed_array);

    return sorted_indices;
}

///////////////////////////////////// SIMPLE LIST OPERATIONS (INTERSECT AND MIRROR) //////////////////////////////////////
// #######################################################################################################################

// Algorithme des deux pointeurs pour l'intersection de deux listes d'entiers, de complexite: O(n+m)
int* intersection(const int* a, int n, const int* b, int m, int *resultSize) {

    int maxSize;
    if (n < m) {
        maxSize = n;
    }
    else{
        maxSize = m;
    }
    int* result = (int*)malloc(maxSize * sizeof(int));

    int i = 0, j = 0, k = 0;

    while (i < n && j < m) {
        if (a[i] == b[j]) {
            result[k++] = a[i];
            i++;
            j++;
        } else if (a[i] < b[j]) {
            i++;
        } else { // a[i] > b[j]
            j++;
        }
    }
    
    *resultSize = k;

    return result;
}

// Return indices i on list a, where a[i] == b[j]
int* arg_intersection_a(const int* a, int n, const int* b, int m, int *resultSize) {

    int maxSize;
    if (n < m) {
        maxSize = n;
    }
    else{
        maxSize = m;
    }
    int* result = (int*)malloc(maxSize * sizeof(int));

    int i = 0, j = 0, k = 0;

    while (i < n && j < m) {
        if (a[i] == b[j]) {
            result[k++] = i;
            i++;
            j++;
        } else if (a[i] < b[j]) {
            i++;
        } else { // a[i] > b[j]
            j++;
        }
    }
    
    *resultSize = k;

    return result;
}

// Function to flip a list
void mirror(int* a, int size) {
    int demi = (int)(size / 2);
    for (int i = 0; i < demi; i++) {
        int val_save = a[i];
        a[i] = a[size-i-1];
        a[size-i-1] = val_save;
    }
}

////////////////////////////////////////////// ORTHONORMALIZATION FUNCTIONS //////////////////////////////////////////////
// #######################################################################################################################

// Gram-Schmidt orthonormalization of V ; returns true iif V is linearly independant
bool orthonormalize(double* V, int rows, int cols, double res) {
    
    if (res < 0) {
        res = 1e-6;
    }

    // Initialize boolean of independancy
    bool independant;

    // Compute max number of linearly independant rows
    int nval;
    if (rows <= cols) {
        nval = rows;
        independant = true;
    }
    else {
        nval = cols;
        independant = false;
    }

    // Dynamically orthonormalize new_V
    for (int i = 0; i < nval; i++) {
        
        double* V_i = V + i * cols;
        
        // compute v - sum( < v , u > . u )_{ for u in U }
        for (int j = 0; j < i; j++) {
            double* V_j = V + j * cols;
            double scal_ij = scalar((const double*)V_j, (const double*)V_i, cols);
            for (int k = 0; k < cols; k++) {
                V_i[k] -= scal_ij * V_j[k];
            }
        }

        // normalize vector new_V_i
        double norm_i = norm((const double*)V_i, cols);
        if (norm_i > res) {
            for (int k = 0; k < cols; k++) {
                V_i[k] /= norm_i;
            }
        }
        else {
            for (int k = 0; k < cols; k++) {
                V_i[k] = 0.0;
            }
            if (independant) {
                independant = false;
            }
            nval++;
        }
    }

    // Put to 0.0 the remaining rows
    for (int i = nval; i < rows; i++) {
        int idx = i * cols;
        int idy = idx + cols;
        for (int j = idx; j < idy; j++) {
            V[j] = 0.0;
        }
    }

    return independant;
}

// it must be guaranteed that, for i = 0 to i = idx-1 included, the V_i are already orthonormalized
bool orthonormalize_from_idx(double* V, int idx, int rows, int cols, double res) {
    
    if (res < 0) {
        res = 1e-6;
    }

    // Initialize boolean of independancy
    bool independant;

    // Compute max number of linearly independant rows
    int nval;
    if (rows <= cols) {
        nval = rows;
        independant = true;
    }
    else if (idx <= cols) {
        nval = cols;
        independant = false;
    }
    else {
        nval = idx;
        independant = false;
    }

    // Dynamically orthonormalize new_V
    for (int i = idx; i < nval; i++) {
        
        double* V_i = V + i * cols;
        
        // compute v - sum( < v , u > . u )_{ for u in U }
        for (int j = 0; j < i; j++) {
            double* V_j = V + j * cols;
            double scal_ij = scalar((const double*)V_j, (const double*)V_i, cols);
            for (int k = 0; k < cols; k++) {
                V_i[k] -= scal_ij * V_j[k];
            }
        }

        // normalize vector new_V_i
        double norm_i = norm((const double*)V_i, cols);
        if (norm_i > res) {
            for (int k = 0; k < cols; k++) {
                V_i[k] /= norm_i;
            }
        }
        else {
            for (int k = 0; k < cols; k++) {
                V_i[k] = 0.0;
            }
            if (independant) {
                independant = false;
            }
            nval++;
        }
    }

    // Put to 0.0 the remaining rows
    for (int i = nval; i < rows; i++) {
        int idx = i * cols;
        int idy = idx + cols;
        for (int j = idx; j < idy; j++) {
            V[j] = 0.0;
        }
    }

    return independant;
}

/////////////////////////////////// FUNCTION TO GET ALL HYPERPLANES CONTAINING POINT q ////////////////////////////////////
// #######################################################################################################################

// Indices of the hyperplanes containing point q
int* contain_q_indices(const double* q, const double* V, const double* s, int rows, int cols, double eps, int *nb_indices) {
    *nb_indices = 0;
    int* indices = (int*)malloc(rows * sizeof(int));
    for (int i = 0; i < rows; i++) {
        double proj_i = scalar(q, V + i * cols, cols) - s[i];
        if (proj_i <= eps && proj_i >= -eps) {
            indices[*nb_indices] = i;
            *nb_indices += 1;
        }
    }
    return indices;
}

/////////////// FUNCTION TO COMPUTE PROXIMITY ERROR (sigma_) AND OPTIMAL MAX NUMBER OF ITERATIONS (big_K) ////////////////
// #######################################################################################################################

// sigma constant
double sigma_(const double* A, const double* b, int rows, int cols) {
    
    double sigma = 0;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double a_ij = A[i * cols + j];
            if (a_ij < 0) {
                a_ij = - a_ij;
            }
            a_ij += 1;
            sigma += log(a_ij);
        }
    }

    if (b != NULL) {
        for (int i = 0; i < rows; i++) {
            double b_i = b[i];
            if (b_i < 0) {
                b_i = - b_i;
            }
            b_i += 1;
            sigma += log(b_i);
        }
    }

    sigma += log(rows * cols);
    sigma += 2;

    return sigma;
}

// max_iter from error computation
int big_K(double error, double lambda, const double* A, const double* b, int rows, int cols) {
    
    if (lambda <= 0 || lambda >= 2) {
        return (int)10000;
    }

    // constants
    double hoffman_constant_2 = 1.5; // Is in [1.0, +inf). Is hardly computable. We hope that 1.5 is enough in most of the cases.
    double sigma = sigma_(A, b, rows, cols);
    //printf("Sigma = %f\n", sigma);

    double max_norm_A = 0.0;
    for (int i = 0; i < rows; i++) {
        double norm_A_i = norm(A + i * cols, cols);
        if (norm_A_i > max_norm_A) {
            max_norm_A = norm_A_i;
        }
    }

    double upper_ln = error * sqrt(cols);
    upper_ln /= max_norm_A;
    upper_ln /= pow(2, 2 * sigma - 2);
    //printf("Upper_ln = %f\n", upper_ln);

    double lower_ln = 2 * lambda - lambda * lambda;
    lower_ln /= rows * hoffman_constant_2 * hoffman_constant_2;
    lower_ln = 1 - lower_ln;
    //printf("Lower_ln = %f\n", lower_ln);

    double k = 2 * log(upper_ln) / log(lower_ln);

    if (!isfinite(k) || (int)k <= 0) {
        fprintf(stderr, "WARNING: no computable big_K. Returned 10000.\n");
        return (int)10000;
    }

    return (int)k;
}

////////////////////////////////// FUNCTION TO GENERATE A LIST OF UNIQUE RANDOM NUMBERS //////////////////////////////////
// #######################################################################################################################

// Random list generation
int* generate_unique_random_numbers(int n, int k) {
    if (k > n) {
        fprintf(stderr, "Erreur : k ne peut pas être supérieur à n.\n");
        exit(EXIT_FAILURE);
    }

    // Créez un tableau contenant tous les entiers de 0 à n
    int *pool = malloc(n * sizeof(int));
    if (!pool) {
        fprintf(stderr, "Erreur : Allocation mémoire échouée.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        pool[i] = i;
    }

    // Mélangez le tableau
    for (int i = 0; i < k; i++) {
        int j = i + rand() % (n - i);
        int temp = pool[i];
        pool[i] = pool[j];
        pool[j] = temp;
    }

    // Copiez les k premiers éléments dans le tableau résultat
    int *result = malloc(k * sizeof(int));
    if (result == NULL) {
        fprintf(stderr, "Erreur : Allocation mémoire échouée.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < k; i++) {
        result[i] = pool[i];
    }

    free(pool);

    return result;
}


////////////////////////////////////////////// LINEAR FEASIBILITY FUNCTIONS //////////////////////////////////////////////
// #######################################################################################################################
// #######################################################################################################################
// #######################################################################################################################

///////////////////////////////////////////////// CONSISTENCY FUNCTIONS //////////////////////////////////////////////////
// #######################################################################################################################

// Function to know if a polyhedron is fully dimensional (exists x such that V * x <= s - res)
bool full_dimensional_GLPK(const double* V, const double* s, int rows, int cols, double res) {
    
    if (res < 0) {
        res = 1e-6;
    }

    glp_prob* lp;

    // Désactiver la sortie de l'optimiseur GLPK (pas d'affichage)
    glp_term_out(GLP_OFF);

    // Etape 1 : Créer un problème GLPK
    lp = glp_create_prob(); // Créer un objet de problème
    glp_set_obj_dir(lp, GLP_MIN);     // Définir la direction de l'optimisation : minimiser

    // Etape 2 : Ajouter les contraintes
    glp_add_rows(lp, rows); // Ajouter les lignes (une par contrainte)
    if (s == NULL) {
        for (int i = 1; i <= rows; i++) {
            glp_set_row_bnds(lp, i, GLP_UP, 0.0, - res); // -A[i].x <= b[i] - res
        }
    }
    else {
        for (int i = 1; i <= rows; i++) {
            glp_set_row_bnds(lp, i, GLP_UP, 0.0, s[i - 1] - res); // -A[i].x <= b[i] - res
        }
    }

    glp_add_cols(lp, cols); // Ajouter les colonnes (une par variable x)
    for (int j = 1; j <= cols; j++) {
        glp_set_col_bnds(lp, j, GLP_FR, 0.0, 0.0); // Pas de bornes sur les variables
        glp_set_obj_coef(lp, j, 0.0);              // Minimiser 0.x
    }

    // Etape 3 : Ajouter les coefficients de la matrice A (en tant que -A pour -Ax <= b)
    int num_elements = rows * cols;
    int* ia = (int*)malloc((num_elements + 1) * sizeof(int)); // Indices des lignes
    int* ja = (int*)malloc((num_elements + 1) * sizeof(int)); // Indices des colonnes
    double* ar = (double*)malloc((num_elements + 1) * sizeof(double)); // Valeurs

    int index = 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ia[index] = i + 1;           // Ligne i+1
            ja[index] = j + 1;           // Colonne j+1
            ar[index] = V[index - 1];  // Coefficient -A[i][j]
            index++;
        }
    }

    glp_load_matrix(lp, num_elements, ia, ja, ar);

    // Etape 4 : Résoudre le problème
    glp_smcp param;
    glp_init_smcp(&param);
    param.presolve = GLP_OFF; // Désactiver la presolve (pour rester fidèle à la version Python)
    int status = glp_simplex(lp, &param);

    // Etape 5 : Vérifier si une solution existe
    bool has_solution = (status == 0 && (glp_get_status(lp) == GLP_FEAS || glp_get_status(lp) == GLP_OPT));

    // Etape 6 : Libérer la mémoire
    glp_delete_prob(lp);
    glp_free_env(); // ???
    
    free(ia);
    free(ja);
    free(ar);

    return has_solution;
}


// Main Sampling Kaczmarz-Motzkin (SKM) algorithm for strict linear feasibility (V_i must all be normed!)
bool full_dimensional_SKM(
    const double* V, const double* s, const double* x0, int rows, int cols, 
    double lambda, int beta, int max_iter, double res
) {

    if (lambda < 0) {
        lambda = 2.0;
    }
    if (beta <= 0) {
        beta = (int)(1.0 * rows);
        if (beta <= 0) {
            beta = 1;
        }
    }
    if (max_iter < 0) {
        max_iter = big_K(1e-3, lambda, V, s, rows, cols);
        if (max_iter > 10000) {
            max_iter = 10000;
        }
    }
    if (res < 0) {
        res = 1e-6;
    }

    double* x = (double*)malloc(cols * sizeof(double));
    if (x0 == NULL) {
        for (int i = 0; i < cols; i++) {
            x[i] = 0.0;
        }
    }
    else {
        for (int i = 0; i < cols; i++) {
            x[i] = x0[i];
        }
    }

    // Adjust s to ~solve V * x < s rather than V * x <= s
    double* new_s = (double*)malloc(rows * sizeof(double));
    if (s == NULL) {
        for (int i = 0; i < rows; i++) {
            new_s[i] = - res;
        }
    }
    else {
        for (int i = 0; i < rows; i++) {
            new_s[i] = s[i] - res;
        }
    }

    double eps = res / (1e1 * rows);

    int k = 0;

    double* proj_qVs = scalar_qVs(V, (const double*)new_s, (const double*)x, rows, cols);
    int arg_max = argmax((const double*)proj_qVs, rows);

    if (beta >= rows) {

        while (proj_qVs[arg_max] > eps && k < max_iter) {
        
            double proj_qAb_max_lambda = lambda * proj_qVs[arg_max];

            const double* a_max = V + arg_max * cols;

            for (int i = 0; i < cols; i++) {
                x[i] = x[i] - proj_qAb_max_lambda * a_max[i];
            }

            for (int i = 0; i < rows; i++) {
                proj_qVs[i] = scalar(x, V + i * cols, cols) - new_s[i];
            }

            arg_max = argmax((const double*)proj_qVs, rows);

            k++;
        }
    }
    else {

        while ((proj_qVs[arg_max] > eps) && (k < max_iter)) {
            
            int* random_I = generate_unique_random_numbers(rows, beta);

            int arg_max_I = argmax_I((const double*)proj_qVs, (const int*)random_I, beta);
            free(random_I);

            double proj_qAb_max_lambda = lambda * proj_qVs[arg_max_I];
            
            if (proj_qAb_max_lambda > 0) {

                const double* a_max = V + arg_max_I * cols;

                for (int i = 0; i < cols; i++) {
                    x[i] = x[i] - proj_qAb_max_lambda * a_max[i];
                }

                for (int i = 0; i < rows; i++) {
                    proj_qVs[i] = scalar(x, V + i * cols, cols) - new_s[i];
                }

                arg_max = argmax((const double*)proj_qVs, rows);
            }

            k++;
        }
    }

    bool state = proj_qVs[arg_max] <= eps;

    free(new_s);
    free(proj_qVs);
    free(x);

    return state;
}


// Sampling Kaczmarz-Motzkin (SKM)-based algorithm for strict linear feasibility (V_i must all be normed!) but adapted to non-orthogonal projections!
bool full_dimensional_Mine(
    const double* V, const double* s, const double* x0, int rows, int cols, 
    double lambda, int max_iter, double res
) {

    // Set undefined constants
    if (lambda < 0) {
        lambda = 1.99;
    }
    int max_reset = (int)1e6; // maximum number of U constraint-resets allowed
    if (max_iter < 0) {
        if (rows > cols + 1) {
            max_iter = 5 * rows; // we let each hyperplane to be considered 5 times in average only
            max_reset = rows;
        }
        else {
            max_iter = 5 * rows; // we let each hyperplane to be considered 2 times in average only
            max_reset = rows;
        }
    }
    if (res < 0) {
        res = 1e-6; // error for strict linear inequality solvation (Ax < b instead of Ax <= b)
    }
    double eps = res / (1e1 * rows); // error for inside-polyhedron estimation (must be < res)
    
    // Initialize dynamical projected point x
    double* x = (double*)malloc(cols * sizeof(double));
    if (x0 == NULL) {
        for (int i = 0; i < cols; i++) {
            x[i] = 0.0;
        }
    }
    else {
        for (int i = 0; i < cols; i++) {
            x[i] = x0[i];
        }
    }

    // Adjust b to ~solve Ax < b instead of Ax <= b
    double* new_s = (double*)malloc(rows * sizeof(double));
    if (s == NULL) {
        for (int i = 0; i < rows; i++) {
            new_s[i] = - res;
        }
    }
    else {
        for (int i = 0; i < rows; i++) {
            new_s[i] = s[i] - res;
        }
    }

    // Initialize dynamical variables
    int nb_iter = 0; // dynamical number of projections of x
    int nb_reset = 0; // dynamical number of U constraint-resets
    bool reset = false;

    // compute orthogonal projections of x on hyperplanes
    double* proj_qVs = scalar_qVs(V, (const double*)new_s, (const double*)x, rows, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows);
    free(proj_qVs);

    // initialize U and rows_U
    double* U = (double*)malloc(cols * cols * sizeof(double));
    int rows_U = 0;

    // Create list of indices I
    int* Id = (int*)malloc(rows * sizeof(int));
    for (int j = 0; j < rows; j++) {
        Id[j] = j;
    }
    int rows_Id = rows;

    // Condition to keep projecting 
    bool q_not_in_psi = max_proj_qVs > eps;

    // We keep projecting x while x is not in Psi
    while (q_not_in_psi && nb_iter < max_iter && nb_reset < max_reset) {

        bool U_isnot_full = rows_U < cols;
        bool h_isnotempty = rows_Id > 0;

        if (U_isnot_full && h_isnotempty) {
        
            // INDEPENDANT: compute W = V_prime
            double* V_prime = (double*)malloc(rows_Id * cols * sizeof(double));
            for (int j = 0; j < rows_Id; j++) {

                // v = V[j]
                int idz = Id[j];

                // compute < v , u > for u in U
                double* scal_v_U = (double*)malloc(rows_U * sizeof(double));
                for (int k = 0; k < rows_U; k++) {
                    scal_v_U[k] = scalar(V + idz * cols, (const double*)U + k * cols, cols);
                }

                // compute v - sum( < v , u > . u )_{ for u in U }
                int rowVp = j * cols;
                int rowV = idz * cols;
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    V_prime[idxVp] = V[rowV + k];
                    for (int l = 0; l < rows_U; l++) {
                        V_prime[idxVp] -= scal_v_U[l] * U[l * cols + k];
                    }
                }

                // free scal_v_U
                free(scal_v_U);
            }

            // norm of V_prime
            double* V_prime_norm = (double*)norm_V((const double*)V_prime, rows_Id, cols);

            // indices of independant vectors v from U
            int num_independant = 0;
            int* idx_independant = (int*)malloc(rows_Id * sizeof(int));
            for (int j = 0; j < rows_Id; j++) {
                if (V_prime_norm[j] > res) {
                    idx_independant[num_independant++] = j;
                }
            }

            // if no independant component, we cannot go further: return q and false
            if (num_independant > 0) {
                
                // POSITIVE: compute < q - C, V > on indices I[idx_original]
                double* proj_qCV = scalar_qVs_IIp(V, (const double*)new_s, (const double*)x, (const int*)Id, (const int*)idx_independant, num_independant, cols);

                // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
                int num_posi_inde = 0;
                int* idx_posi_inde = (int*)malloc(num_independant * sizeof(int));
                int* idx_is_posi__ = (int*)malloc(num_independant * sizeof(int));
                for (int j = 0; j < num_independant; j++) {
                    if (proj_qCV[j] > eps) {
                        idx_posi_inde[num_posi_inde] = idx_independant[j];
                        idx_is_posi__[num_posi_inde] = j;
                        num_posi_inde++;
                    }
                }

                // if no positive AND independant component, we choose to not go further: return q and false
                if (num_posi_inde > 0) {
                    
                    // normed V_prime on num_posi_inde
                    double* V_prime_normed = (double*)malloc(num_posi_inde * cols * sizeof(double));
                    for (int j = 0; j < num_posi_inde; j++) {
                        int idx = idx_posi_inde[j]; // idx_posi_inde[j] = idx_independant[idx_is_inde__[j]]
                        int rowVpn = j * cols;
                        int rowVp = idx * cols;
                        for (int k = 0; k < cols; k++) {
                            V_prime_normed[rowVpn + k] = V_prime[rowVp + k] / V_prime_norm[idx];
                        }
                    }
                    free(V_prime);
                    free(V_prime_norm);

                    // compute projection distances
                    double* distances = (double*)malloc(num_posi_inde * sizeof(double));
                    for (int j = 0; j < num_posi_inde; j++) {
                        int idx = Id[idx_posi_inde[j]]; // idx_posi_inde[j] = idx_independant[idx_is_inde__[j]]
                        int idy = idx_is_posi__[j];
                        distances[j] = proj_qCV[idy] / scalar(V + idx * cols, (const double*)V_prime_normed + j * cols, cols);
                    }
                    free(proj_qCV);
                    free(idx_is_posi__);

                    // Get max of distances distances!
                    int idx_max = argmax((const double*)distances, num_posi_inde);
                    double max_lambda_distance = lambda * distances[idx_max];
                    free(distances);

                    // computing I_p
                    for (int j = 0; j < num_independant; j++) {
                        Id[j] = Id[idx_independant[j]];
                    }
                    rows_Id = num_independant;
                    free(idx_independant);
                    free(idx_posi_inde);

                    // project x
                    double* Vpn = V_prime_normed + idx_max * cols;
                    for (int j = 0; j < cols; j++) {
                        x[j] = x[j] - max_lambda_distance * Vpn[j];
                    }

                    // update U
                    double* U_i = U + rows_U * cols;
                    for (int j = 0; j < cols; j++) {
                        U_i[j] = Vpn[j];
                    }
                    rows_U++;

                    free(V_prime_normed);

                }
                else {
                    free(V_prime);
                    free(V_prime_norm);
                    free(idx_independant);
                    
                    free(proj_qCV);
                    free(idx_posi_inde);
                    free(idx_is_posi__);

                    reset = true;
                }
            }
            else {
                free(V_prime);
                free(V_prime_norm);
                free(idx_independant);

                reset = true;
            }
        }
        else {
            reset = true;
        }

        if (reset) {

            // reset U
            rows_U = 0;

            // reset I
            for (int j = 0; j < rows; j++) {
                Id[j] = j;
            }
            rows_Id = rows;

            // no more reset needed
            reset = false;

            // one reset had been made
            nb_reset++;
        }
        else {

            // compute orthogonal projections of x on all hyperplanes
            double* proj_qAb = scalar_qVs(V, (const double*)new_s, (const double*)x, rows, cols);
            double max_proj_qAb = max((const double*)proj_qAb, rows);
            free(proj_qAb);

            // check if x is in polyhedron Psi
            q_not_in_psi = max_proj_qAb > eps;

            // one projection had been made
            nb_iter++;
        }
    }

    free(U);
    free(Id);

    free(new_s);
    free(x);

    return !q_not_in_psi;
}


// Main function to know a polyedron defined by (V,s) is full-dimensional or not (3 methods: GLPK, SKM, Mine)
bool full_dimensional(const double* V, const double* s, int rows, int cols, double res) {

    int method = 1;

    if (method == 1) {
        return full_dimensional_GLPK(V, s, rows, cols, res);
    }
    else if (method == 2) {
        return full_dimensional_SKM (V, s, NULL, rows, cols, -1.0, -1, -1, res);
    }
    else {
        return full_dimensional_Mine(V, s, NULL, rows, cols, -1.0, -1, res);
    }
}


//////////////////////////////////////////////////// MINOR FUNCTIONS /////////////////////////////////////////////////////
// #######################################################################################################################

// Function to know if q is minor regarding p in polyhedron (V,s) (q is supposed to be in it, and p not) - with GLPK feasibilty
bool minor_GLPK(const double* q, const double* p, const double* V, const double* s, int rows, int cols, bool q_inside, double res) {

    if (res < 0) {
        res = 1e-6;
    }

    // Compute v0 = normed(q-p)
    double* v0 = (double*)malloc(cols * sizeof(double));
    if (p == NULL) {
        double v0_norm = norm(q, cols);
        if (v0_norm <= res) {
            free(v0);
            return true;
        }
        for (int i = 0; i < cols; i++) {
            v0[i] = q[i] / v0_norm;
        }
    }
    else {
        for (int i = 0; i < cols; i++) {
            v0[i] = q[i] - p[i];
        }
        double v0_norm = norm((const double*)v0, cols);
        if (v0_norm <= res) {
            free(v0);
            return true;
        }
        for (int i = 0; i < cols; i++) {
            v0[i] /= v0_norm;
        }
    }

    // Evaluate if the new Polyhedron is empty or not
    bool result;
    if (q_inside) { // q is inside the polyhedron described by (V,s)

        // Compute the indices on (V,s) for which the associated hyperplanes contain q
        int nb_indices;
        int* indices = contain_q_indices(q, V, s, rows, cols, res, &nb_indices);

        // If no hyperplane contains q, then q is necessarily not minor
        if (nb_indices == 0) {
            result = false;
        }
        else {

            int new_rows = nb_indices + 1;
            
            // Copy and norm V in new_V on the indices
            double* new_V = (double*)malloc(new_rows * cols * sizeof(double));
            for (int i = 0; i < nb_indices; i++) {
                
                double* new_V_i = new_V + i * cols;
                const double* V_i = V + indices[i] * cols;
                
                double norm_i = norm(V_i, cols);

                if (norm_i <= res) {
                    for (int j = 0; j < cols; j++) {
                        new_V_i[j] = 0.0;
                    }
                }
                else {
                    for (int j = 0; j < cols; j++) {
                        new_V_i[j] = V_i[j] / norm_i;
                    }
                }
            }
            
            // Add v0 to new_V
            double* new_V_i = new_V + nb_indices * cols;
            for (int j = 0; j < cols; j++) {
                new_V_i[j] = v0[j];
            }

            // Result with GLPK method
            result = !full_dimensional_GLPK((const double*)new_V, NULL, new_rows, cols, res);

            free(new_V);
        }

        free(indices);
    }
    else { // q is outside the polyhedron described by (V,s)
        
        int new_rows = rows + 1;
        
        // Copy and norm V in new_V
        double* new_V = (double*)malloc(new_rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            
            double* new_V_i = new_V + i * cols;
            const double* V_i = V + i * cols;
            
            double norm_i = norm(V_i, cols);

            if (norm_i <= res) {
                for (int j = 0; j < cols; j++) {
                    new_V_i[j] = 0.0;
                }
            }
            else {
                for (int j = 0; j < cols; j++) {
                    new_V_i[j] = V_i[j] / norm_i;
                }
            }
        }
        
        // Add v0 to new_V
        double* new_V_i = new_V + rows * cols;
        for (int j = 0; j < cols; j++) {
            new_V_i[j] = v0[j];
        }

        // Copy s in new_s
        double* new_s = (double*)malloc(new_rows * sizeof(double));
        for (int i = 0; i < rows; i++) {
            new_s[i] = s[i];
        }

        // Add <v0,q> to new_s
        new_s[rows] = scalar((const double*)v0, q, cols);

        // Result with GLPK method
        result = !full_dimensional_GLPK((const double*)new_V, (const double*)new_s, new_rows, cols, res);

        free(new_V);
        free(new_s);
    }

    free(v0);

    return result;
}


// Function to know if q is minor regarding p in polyhedron (V,s) (q is supposed to be in it, and p not) - with SKM algorithm
bool minor_SKM(const double* q, const double* p, const double* V, const double* s, int rows, int cols, bool q_inside, double res) {

    if (res < 0) {
        res = 1e-6;
    }

    // Compute v0 = normed(q-p)
    double* v0 = (double*)malloc(cols * sizeof(double));
    if (p == NULL) {
        double v0_norm = norm(q, cols);
        if (v0_norm <= res) {
            free(v0);
            return true;
        }
        for (int i = 0; i < cols; i++) {
            v0[i] = q[i] / v0_norm;
        }
    }
    else {
        for (int i = 0; i < cols; i++) {
            v0[i] = q[i] - p[i];
        }
        double v0_norm = norm((const double*)v0, cols);
        if (v0_norm <= res) {
            free(v0);
            return true;
        }
        for (int i = 0; i < cols; i++) {
            v0[i] /= v0_norm;
        }
    }

    // Evaluate if the new Polyhedron is empty or not
    bool result;
    if (q_inside) { // q is inside the polyhedron described by (V,s)

        // Compute the indices on (V,s) for which the associated hyperplanes contain q
        int nb_indices;
        int* indices = contain_q_indices(q, V, s, rows, cols, res, &nb_indices);

        // If no hyperplane contains q, then q is necessarily not minor
        if (nb_indices == 0) {
            result = false;
        }
        else {

            int new_rows = nb_indices + 1;
            
            // Copy and norm V in new_V on the indices
            double* new_V = (double*)malloc(new_rows * cols * sizeof(double));
            for (int i = 0; i < nb_indices; i++) {
                
                double* new_V_i = new_V + i * cols;
                const double* V_i = V + indices[i] * cols;
                
                double norm_i = norm(V_i, cols);

                if (norm_i <= res) {
                    for (int j = 0; j < cols; j++) {
                        new_V_i[j] = 0.0;
                    }
                }
                else {
                    for (int j = 0; j < cols; j++) {
                        new_V_i[j] = V_i[j] / norm_i;
                    }
                }
            }
            
            // Add v0 to new_V
            double* new_V_i = new_V + nb_indices * cols;
            for (int j = 0; j < cols; j++) {
                new_V_i[j] = v0[j];
            }

            // Result with SKM method
            result = !full_dimensional_SKM((const double*)new_V, NULL, NULL, new_rows, cols, -1.0, -1, -1, res);

            free(new_V);
        }

        free(indices);
    }
    else { // q is outside the polyhedron described by (V,s)
        
        int new_rows = rows + 1;
        
        // Copy and norm V in new_V
        double* new_V = (double*)malloc(new_rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            
            double* new_V_i = new_V + i * cols;
            const double* V_i = V + i * cols;
            
            double norm_i = norm(V_i, cols);

            if (norm_i <= res) {
                for (int j = 0; j < cols; j++) {
                    new_V_i[j] = 0.0;
                }
            }
            else {
                for (int j = 0; j < cols; j++) {
                    new_V_i[j] = V_i[j] / norm_i;
                }
            }
        }
        
        // Add v0 to new_V
        double* new_V_i = new_V + rows * cols;
        for (int j = 0; j < cols; j++) {
            new_V_i[j] = v0[j];
        }

        // Copy s in new_s
        double* new_s = (double*)malloc(new_rows * sizeof(double));
        for (int i = 0; i < rows; i++) {
            new_s[i] = s[i];
        }

        // Add <v0,q> to new_s
        new_s[rows] = scalar((const double*)v0, q, cols);

        // Result with SKM method
        result = !full_dimensional_SKM((const double*)new_V, (const double*)new_s, q, new_rows, cols, -1.0, -1, -1, res);

        free(new_V);
        free(new_s);
    }

    free(v0);

    return result;
}


// Function to know if q is minor regarding p in polyhedron (V,s) (q is supposed to be in it, and p not) - with Mine algorithm (SKM-like)
bool minor_Mine(const double* q, const double* p, const double* V, const double* s, int rows, int cols, bool q_inside, double res) {

    if (res < 0) {
        res = 1e-6;
    }

    // Compute v0 = normed(q-p)
    double* v0 = (double*)malloc(cols * sizeof(double));
    if (p == NULL) {
        double v0_norm = norm(q, cols);
        if (v0_norm <= res) {
            free(v0);
            return true;
        }
        for (int i = 0; i < cols; i++) {
            v0[i] = q[i] / v0_norm;
        }
    }
    else {
        for (int i = 0; i < cols; i++) {
            v0[i] = q[i] - p[i];
        }
        double v0_norm = norm((const double*)v0, cols);
        if (v0_norm <= res) {
            free(v0);
            return true;
        }
        for (int i = 0; i < cols; i++) {
            v0[i] /= v0_norm;
        }
    }

    // Evaluate if the new Polyhedron is empty or not
    bool result;
    if (q_inside) { // q is inside the polyhedron described by (V,s)

        // Compute the indices on (V,s) for which the associated hyperplanes contain q
        int nb_indices;
        int* indices = contain_q_indices(q, V, s, rows, cols, res, &nb_indices);

        // If no hyperplane contains q, then q is necessarily not minor
        if (nb_indices == 0) {
            result = false;
        }
        else {

            int new_rows = nb_indices + 1;
            
            // Copy and norm V in new_V on the indices
            double* new_V = (double*)malloc(new_rows * cols * sizeof(double));
            for (int i = 0; i < nb_indices; i++) {
                
                double* new_V_i = new_V + i * cols;
                const double* V_i = V + indices[i] * cols;
                
                double norm_i = norm(V_i, cols);

                if (norm_i <= res) {
                    for (int j = 0; j < cols; j++) {
                        new_V_i[j] = 0.0;
                    }
                }
                else {
                    for (int j = 0; j < cols; j++) {
                        new_V_i[j] = V_i[j] / norm_i;
                    }
                }
            }
            
            // Add v0 to new_V
            double* new_V_i = new_V + nb_indices * cols;
            for (int j = 0; j < cols; j++) {
                new_V_i[j] = v0[j];
            }

            // Result with Mine method
            result = !full_dimensional_Mine((const double*)new_V, NULL, NULL, new_rows, cols, -1.0, -1, res);

            free(new_V);
        }

        free(indices);
    }
    else { // q is outside the polyhedron described by (V,s)
        
        int new_rows = rows + 1;
        
        // Copy and norm V in new_V
        double* new_V = (double*)malloc(new_rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            
            double* new_V_i = new_V + i * cols;
            const double* V_i = V + i * cols;
            
            double norm_i = norm(V_i, cols);

            if (norm_i <= res) {
                for (int j = 0; j < cols; j++) {
                    new_V_i[j] = 0.0;
                }
            }
            else {
                for (int j = 0; j < cols; j++) {
                    new_V_i[j] = V_i[j] / norm_i;
                }
            }
        }
        
        // Add v0 to new_V
        double* new_V_i = new_V + rows * cols;
        for (int j = 0; j < cols; j++) {
            new_V_i[j] = v0[j];
        }

        // Copy s in new_s
        double* new_s = (double*)malloc(new_rows * sizeof(double));
        for (int i = 0; i < rows; i++) {
            new_s[i] = s[i];
        }

        // Add <v0,q> to new_s
        new_s[rows] = scalar((const double*)v0, q, cols);

        // Result with Mine method
        result = !full_dimensional_Mine((const double*)new_V, (const double*)new_s, q, new_rows, cols, -1.0, -1, res);

        free(new_V);
        free(new_s);
    }

    free(v0);

    return result;
}


// Function to know if q is minor regarding p in polyhedron (V,s) (q is supposed to be in it, and p not) - pure geometry building ortho. polyhedral cone
bool minor_Cone(const double* q, const double* p, const double* V, const double* s, int rows, int cols, double res) {
    
    if (res < 0) {
        res = 1e-6;
    }

    // Compute v0 = normed(q-p)
    double* v0 = (double*)malloc(cols * sizeof(double));
    if (p == NULL) {
        double v0_norm = norm(q, cols);
        if (v0_norm <= res) {
            free(v0);
            return true;
        }
        for (int i = 0; i < cols; i++) {
            v0[i] = q[i] / v0_norm;
        }
    }
    else {
        for (int i = 0; i < cols; i++) {
            v0[i] = q[i] - p[i];
        }
        double v0_norm = norm((const double*)v0, cols);
        if (v0_norm <= res) {
            free(v0);
            return true;
        }
        for (int i = 0; i < cols; i++) {
            v0[i] /= v0_norm;
        }
    }

    // Compute the indices on (V,s) for which the associated hyperplanes contain q
    int nb_indices;
    int* indices = contain_q_indices(q, V, s, rows, cols, res, &nb_indices);

    // Evaluate if the new Polyhedron is empty or not
    bool result;
    if (nb_indices == 0) {
        result = false;
    }
    else if (nb_indices > cols) {

        //TODO: 2nd method

        printf("WARNING: 2nd method (minor_Mine) used (the rows V_i are not linearly independant)!\n");
        result = minor_Mine(q, p, V, s, rows, cols, true, res);

    }
    else {

        // Copy V in new_V
        double* new_V = (double*)malloc(nb_indices * cols * sizeof(double));
        for (int i = 0; i < nb_indices; i++) {
            double* new_V_i = new_V + i * cols;
            const double* V_i = V + indices[i] * cols;
            for (int j = 0; j < cols; j++) {
                new_V_i[j] = V_i[j];
            }
        }

        // Orthonormalize new_V and check linear dependancy
        bool is_linearly_independant = orthonormalize(new_V, nb_indices, cols, res);

        if (!is_linearly_independant) {

            //TODO: 2nd method

            printf("WARNING: 2nd method (minor_Mine) used (the rows V_i are not linearly independant)!\n");
            result = minor_Mine(q, p, V, s, rows, cols, true, res);

        }
        else{

            bool v0_in_Vspace;

            if (nb_indices == cols) {
                v0_in_Vspace = true;
            }
            else {
                double* v0_onVT = (double*)malloc(cols * sizeof(double));
                for (int i = 0; i < cols; i++) {
                    v0_onVT[i] = v0[i];
                }
                for (int i = 0; i < nb_indices; i++) {
                    double* new_V_i = new_V + i * cols;
                    double scal_i = scalar((const double*)v0_onVT, (const double*)new_V_i, cols);
                    for (int k = 0; k < cols; k++) {
                        v0_onVT[k] -= scal_i * new_V_i[k];
                    }
                }

                double norm_v0_on_VT = norm((const double*)v0_onVT, cols);
                free(v0_onVT);

                v0_in_Vspace = norm_v0_on_VT <= res;
            }

            if (v0_in_Vspace) {

                int last_idx = nb_indices - 1;

                // check if v0 is behind the last vector from new_V
                double* new_V_last = new_V + last_idx * cols;
                result = scalar((const double*)v0, (const double*)new_V_last, cols) <= res;

                if (result) {

                    // building the other vectors successively
                    for (int i = last_idx - 1; i >= 0; i--) {

                        // swap indices i and last_idx
                        int idx_save = indices[i];
                        indices[i] = indices[last_idx];
                        indices[last_idx] = idx_save;
                        for (int j = i; j < nb_indices; j++) {
                            double* new_V_j = new_V + j * cols;
                            const double* V_id_j = V + indices[j] * cols;
                            for (int k = 0; k < cols; k++) {
                                new_V_j[k] = V_id_j[k];
                            }
                        }

                        // orthonormalize new_V from index i
                        orthonormalize_from_idx(new_V, i, nb_indices, cols, res);

                        // check if v0 is behind the last vector from new_V
                        double* new_V_last = new_V + last_idx * cols;
                        result = scalar((const double*)v0, (const double*)new_V_last, cols) <= res;

                        // continue only if result is true
                        if (!result) {
                            break;
                        }
                    }
                }
            }
            else {
                result = false;
            }            
        }

        free(new_V);
    }

    free(v0);
    free(indices);

    return result;
}


// Main function to know if q is minor regarding p in polyhedron (V,s) (q is supposed to be in it, and p not) (4 methods: GLPK, SKM, Mine, Cone)
bool minor(const double* q, const double* p, const double* V, const double* s, int rows, int cols, bool q_inside, double res) {

    int method;
    if(q_inside) {
        method = 4;
    }
    else {
        method = 1;
    }

    if (method == 1) {
        return minor_GLPK(q, p, V, s, rows, cols, q_inside, res);
    }
    else if (method == 2) {
        return minor_SKM (q, p, V, s, rows, cols, q_inside, res);
    }
    else if (method == 3) {
        return minor_Mine(q, p, V, s, rows, cols, q_inside, res);
    }
    else {
        return minor_Cone(q, p, V, s, rows, cols, res); // q must necessarily be inside (q is the origin of the cone)
    }
}


////////////////////////////////////////////////// NECESSITY FUNCTIONS ///////////////////////////////////////////////////
// #######################################################################################################################

// Function to know if a vector couple is necessary
//TODO: try to directly modify V,s (NO COPY - malloc)
bool couple_is_necessary(int i, double* V, double* s, int rows, int cols, double res) {
    
    double* V_i = V + i * cols;

    // Etape 1 : allouer - V[i]
    s[i] = - s[i];
    for (int k = 0; k < cols; k++) {
        V_i[k] = - V_i[k];
    }

    // Etape 2 : Vérifier si le polyèdre est pleinement dimensionnel
    bool result = full_dimensional((const double*)V, (const double*)s, rows, cols, res);

    // Etape 3: Réinverser
    s[i] = - s[i];
    for (int k = 0; k < cols; k++) {
        V_i[k] = - V_i[k];
    }

    return result;
}

// Function to keep only necessary couples
void keep_only_necessary_couples(const double* V, const double* s, int rows, int cols, double res, double **out_V, double **out_s, int *out_rows) {
    
    int new_rows = rows;

    // Copy V and s in new_V and new_s
    double* new_V = (double*)malloc(rows * cols * sizeof(double));
    double* new_s = (double*)malloc(rows * sizeof(double));
    for (int j = 0; j < rows; j++) {
        new_s[j] = s[j];
        int row = j * cols;
        for (int k = 0; k < cols; k++) {
            int idx = row + k;
            new_V[idx] = V[idx];
        }
    }

    int i = rows - 1;
    while (i >= 0) {
        bool i_is_neessary = couple_is_necessary(i, new_V, new_s, new_rows, cols, res);
        if (!i_is_neessary) {
            
            // Update new_V and new_s
            for (int j = i; j < new_rows - 1; j++) {
                new_s[j] = new_s[j+1];
                int rowA = j * cols;
                int rowB = rowA + cols;
                for (int k = 0; k < cols; k++) {
                    new_V[rowA + k] = new_V[rowB + k];
                }
            }

            new_rows --;
        }
        i --;
    }

    *out_V = new_V;
    *out_s = new_s;
    *out_rows = new_rows;
}

// Function to keep only necessary couples (directly modifies V and s, and returns out_rows)
int keep_only_necessary_couples_ModifyMatrices(double* V, double* s, int rows, int cols, double res) {
    
    int new_rows = rows;

    int i = rows - 1;
    while (i >= 0) {
        bool i_is_neessary = couple_is_necessary(i, V, s, new_rows, cols, res);
        if (!i_is_neessary) {
            
            // Update new_V and new_s
            for (int j = i; j < new_rows - 1; j++) {
                s[j] = s[j+1];
                int rowA = j * cols;
                int rowB = rowA + cols;
                for (int k = 0; k < cols; k++) {
                    V[rowA + k] = V[rowB + k];
                }
            }

            new_rows --;
        }
        i --;
    }

    return new_rows;
}

// Function to get the indices of necessary couples only
int* keep_only_necessary_couples_idx(const double* V, const double* s, int rows, int cols, double res, int *out_rows) {
    
    int new_rows = rows;

    int* idx = (int*)malloc(rows * sizeof(int));

    // Copy V and s in new_V and new_s
    double* new_V = (double*)malloc(rows * cols * sizeof(double));
    double* new_s = (double*)malloc(rows * sizeof(double));
    for (int j = 0; j < rows; j++) {
        new_s[j] = s[j];
        int row = j * cols;
        for (int k = 0; k < cols; k++) {
            int idx = row + k;
            new_V[idx] = V[idx];
        }
    }

    int i = rows - 1;
    while (i >= 0) {
        bool i_is_neessary = couple_is_necessary(i, new_V, new_s, new_rows, cols, res);
        if (!i_is_neessary) {
            
            // Update new_V and new_s
            for (int j = i; j < new_rows - 1; j++) {
                new_s[j] = new_s[j+1];
                int rowA = j * cols;
                int rowB = rowA + cols;
                for (int k = 0; k < cols; k++) {
                    new_V[rowA + k] = new_V[rowB + k];
                }
            }

            new_rows --;
        }
        else {
            idx[new_rows - i - 1] = i;
        }
        i --;
    }

    free(new_V);
    free(new_s);

    *out_rows = new_rows;

    return idx;
}

// Function to get the indices of necessary couples only - Warning: Id must be sorted (from smallest to biggest indices)!!!
int* keep_only_necessary_couples_idx_I(const double* V, const double* s, int rows, const int* Id, int rows_Id, int cols, double res, int *out_rows) {
    
    int new_rows = rows;
    int new_rows_Id = rows_Id;

    int* idx = (int*)malloc(rows_Id * sizeof(int));

    // Copy V and s in new_V and new_s
    double* new_V = (double*)malloc(rows * cols * sizeof(double));
    double* new_s = (double*)malloc(rows * sizeof(double));
    for (int j = 0; j < rows; j++) {
        new_s[j] = s[j];
        int row = j * cols;
        for (int k = 0; k < cols; k++) {
            int idx = row + k;
            new_V[idx] = V[idx];
        }
    }

    int i = rows_Id - 1;
    while (i >= 0) {
        int id = Id[i];
        bool i_is_neessary = couple_is_necessary(id, new_V, new_s, new_rows, cols, res);
        if (!i_is_neessary) {
            
            // Update new_V and new_s
            for (int j = id; j < new_rows - 1; j++) {
                new_s[j] = new_s[j+1];
                int rowA = j * cols;
                int rowB = rowA + cols;
                for (int k = 0; k < cols; k++) {
                    new_V[rowA + k] = new_V[rowB + k];
                }
            }

            new_rows --;
            new_rows_Id --;
        }
        else {
            idx[new_rows_Id - i - 1] = id;
        }
        i --;
    }

    free(new_V);
    free(new_s);

    *out_rows = new_rows_Id;

    return idx;
}


////////////////////////////////////// MAIN FUNCTIONS TO COMPUTE MINIMUM-NORM POINT //////////////////////////////////////
// #######################################################################################################################
// #######################################################################################################################
// #######################################################################################################################

///////////////////// FIRST VERSION: SIMPLIEST, BUT TAKING ARGMAX ONLY WHEN REMAINING DIMENSION IS 1 /////////////////////
// #######################################################################################################################

// f0 recursive function
bool project_q_0dr1(
    double* q, int cols, 
    const double* V, const double* s, int rows_Vs, 
    const double* W, const double* z, int rows_Wz, 
    const double* u, int depth, 
    const double res, int *nb_recursions
) {
    *nb_recursions += 1;

    // Compute the whole < q , V > - s and its maximum
    double* proj_qVs = scalar_qVs(V, s, (const double*)q, rows_Vs, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows_Vs);
    free(proj_qVs);

    // Three conditions to keep projecting q
    bool q_not_in_psi = max_proj_qVs > res;
    bool U_isnot_full = depth < cols;
    bool h_isnotempty = rows_Wz > 0;

    // We keep projecting q
    if (q_not_in_psi && U_isnot_full && h_isnotempty) {

        // INDEPENDANT: compute W_prime
        double* W_prime = (double*)malloc(rows_Wz * cols * sizeof(double));
        if (u != NULL) { //TODO: WHAT IF u IS NULL???
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute < w , u > for u = U[-1]
                double scal_vu = scalar(W + rowVp, u, cols);

                // compute w - < w , u > . u for u = U[-1]
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp] - scal_vu * u[k];
                }
            }
        }
        else {
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute w_prime = w
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp];
                }
            }
        }

        // norm of W_prime
        double* W_prime_norm = (double*)norm_V((const double*)W_prime, rows_Wz, cols);

        // indices of independant vectors v from U
        int num_independant = 0;
        int* idx_independant = (int*)malloc(rows_Wz * sizeof(int));
        for (int j = 0; j < rows_Wz; j++) {
            if (W_prime_norm[j] > res) {
                idx_independant[num_independant++] = j;
            }
        }

        // if no independant component, we cannot go further: return q and false
        if (num_independant == 0) {

            free(W_prime);
            free(W_prime_norm);
            free(idx_independant);

            return false;
        }

        // POSITIVE: compute < q - X, W > on indices idx_independant
        double* proj_qWz = scalar_qVs_I(W, z, (const double*)q, (const int*)idx_independant, num_independant, cols);

        // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
        int num_posi_inde = 0;
        int* idx_is_posi__ = (int*)malloc(num_independant * sizeof(int));
        int* idx_is_nega__ = (int*)malloc(num_independant * sizeof(int));
        for (int j = 0; j < num_independant; j++) {
            if (proj_qWz[j] > res) {
                idx_is_posi__[num_posi_inde] = j;
                num_posi_inde++;
            }
            else {
                idx_is_nega__[j - num_posi_inde] = j;
            }
        }

        // if no positive AND independant component, we choose to not go further: return q and false
        if (num_posi_inde == 0) {

            free(W_prime);
            free(W_prime_norm);
            free(idx_independant);

            free(proj_qWz);
            free(idx_is_posi__);
            free(idx_is_nega__);

            return false;
        }

        // save a copy of the current q in new vector q_p
        double* q_p = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            q_p[j] = q[j];
        }

        // initialize state to false
        bool state = false;

        // !!!!! HERE for 1D or NOT !!!!!
        if (cols - depth == 1) { // remaining_projection_dimension is == 1

            free(idx_is_nega__);

            // normed V_prime on num_posi_inde - ON POSITIVE COMPONENTS ONLY! (unnecessary to do it on negative ones!)
            double* W_prime_normed = (double*)malloc(num_posi_inde * cols * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_independant[idx_is_posi__[j]];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_posi_inde * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_is_posi__[j];
                distances[j] = proj_qWz[k] / scalar(W + idx_independant[k] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(idx_is_posi__);
            free(proj_qWz);

            // compute argmax of distances
            int arg_max = argmax((const double*)distances, num_posi_inde);

            // update q (projected on furthest hyperplane)
            double dis_i = distances[arg_max];
            double* vec_i = W_prime_normed + arg_max * cols;
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j] - dis_i * vec_i[j];
            }
            free(W_prime_normed);
            free(distances);

            // call function project_q (go deeper in projections)
            state = project_q_0dr1(
                q, cols, 
                V, s, rows_Vs, 
                NULL, NULL, 0, 
                NULL, depth + 1, 
                res, nb_recursions
            ); // q is updated

        }
        else {

            // normed V_prime on num_independant
            double* W_prime_normed = (double*)malloc(num_independant * cols * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                int k = idx_independant[j];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                distances[j] = proj_qWz[j] / scalar(W + idx_independant[j] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(proj_qWz);

            // Sort distances! (on idx_is_posi__) - local indices for idx_independant
            int* idx_order = (int*)argsort_desc_I((const double*)distances, idx_is_posi__, num_posi_inde);
            free(idx_is_posi__);

            // compute new vectors NEW_W and new points NEW_X
            double* NEW_W = (double*)malloc(num_independant * cols * sizeof(double));
            double* NEW_z = (double*)malloc(num_independant * sizeof(double));
            double* NEW_distances = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant - num_posi_inde; j++) { // first, negative
                int idz = idx_is_nega__[j];
                int rowJj = j * cols;
                int rowJn = idz * cols;
                NEW_z[j] = scalar((const double*)q, (const double*)W_prime_normed + rowJn, cols) - distances[idz];
                NEW_distances[j] = distances[idz];
                for (int k = 0; k < cols; k++) {
                    NEW_W[rowJj + k] = W_prime_normed[rowJn + k];
                }
            }
            free(idx_is_nega__);
            for (int j = 0; j < num_posi_inde; j++) { // then, positive, sorted (ascending)
                int l = num_independant - j - 1;
                int idz = idx_order[j];
                int rowJl = l * cols;
                int rowJn = idz * cols;
                NEW_z[l] = scalar((const double*)q, (const double*)W_prime_normed + rowJn, cols) - distances[idz];
                NEW_distances[l] = distances[idz];
                for (int k = 0; k < cols; k++) {
                    NEW_W[rowJl + k] = W_prime_normed[rowJn + k];
                }
            }
            free(W_prime_normed);
            free(idx_order);
            free(distances);
            
            // Project on new hyperplane
            int i = 0;
            while (!state && i < num_posi_inde) {

                // numeral of projection
                int num_proj = num_independant - i - 1;

                // direction of projection is the vector of the furthest hyperplane
                double* NEW_u = NEW_W + num_proj * cols;

                // update q (projected on furthest hyperplane)
                double dis_i = NEW_distances[num_proj];
                for (int j = 0; j < cols; j++) {
                    q[j] = q_p[j] - dis_i * NEW_u[j];
                }

                // call function project_q (go deeper in projections)
                state = project_q_0dr1(
                    q, cols, 
                    V, s, rows_Vs, 
                    (const double*)NEW_W, (const double*)NEW_z, num_proj, 
                    (const double*)NEW_u, depth + 1, 
                    res, nb_recursions
                ); // q is dynamically updated

                i++;
            }

            free(NEW_W);
            free(NEW_z);
            free(NEW_distances);
        }

        if (state) {
            free(q_p);
            return true;
        }
        else {
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j];
            }
            free(q_p);
            return false;
        }
    }
    else if (q_not_in_psi) { // q is not in Psi AND we cannot project deeper!
        return false;
    }
    else if (depth <= 1) { // q is necessarily minor, because only projected on one hyperplane at max!
        return true;
    }
    else if (!minor((const double*)q, NULL, V, s, rows_Vs, cols, true, 1e-6)) { // q is in Psi but is not minor // is 1e-6 a good res???
        return false;
    }
    else { // q is in Psi AND is minor
        return true;
    }
}


double* algo_0dr1(const double* V, const double* s, int rows, int cols, const double* p, const double res) {

    double* proj_qVs = scalar_qVs(V, s, p, rows, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows);
    free(proj_qVs);

    // If point p is in polyhedron, then return copy of p
    if (max_proj_qVs <= res) {
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = p[k];
        }
        return q;
    }
    else {
        // initialize q
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = 0.0;
        }

        // Substract <p,v> to s
        double* s_p = (double*)malloc(rows * sizeof(double));
        for (int k = 0; k < rows; k++) {
            s_p[k] = s[k] - scalar(p, V + k * cols, cols);
        }

        // initialize depth
        int depth = 0;
        int nb_recursions = -1;

        // main algo to project iteratively q on polyhedron
        bool state = project_q_0dr1(
            q, cols, 
            (const double*)V, (const double*)s_p, rows, 
            (const double*)V, (const double*)s_p, rows, 
            NULL, depth, 
            res, &nb_recursions
        ); // q is dynamically updated

        //printf("%i ",nb_recursions);

        // free s_p
        free(s_p);

        // add p back to q
        for (int k = 0; k < cols; k++) {
            q[k] = (double)nb_recursions;// += p[k];
        }

        // If no minimum-norm point found, print warning message!
        if (!state){
            printf("WARNING: no minimum-norm point found!\n");
        }

        return q;
    }
}







///////////////////// Third VERSION: SIMPLIEST, BUT TAKING ARGMAX ONLY WHEN REMAINING DIMENSION IS 1 /////////////////////
// #######################################################################################################################

// f0 recursive function
bool project_q_2dr1(
    double* q, int cols, 
    const double* V, const double* s, int rows_Vs, 
    const double* W, const double* z, int rows_Wz, 
    const double* u, int depth, 
    const double res, int *nb_recursions
) {
    *nb_recursions += 1;

    // Compute the whole < q , V > - s and its maximum
    double* proj_qVs = scalar_qVs(V, s, (const double*)q, rows_Vs, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows_Vs);
    free(proj_qVs);

    // Three conditions to keep projecting q
    bool q_not_in_psi = max_proj_qVs > res;
    bool U_isnot_full = depth < cols;
    bool h_isnotempty = rows_Wz > 0;

    bool q_is_minor;
    if (depth <= 1) {
        q_is_minor = true;
    }
    else {
        if (!q_not_in_psi) {
            q_is_minor = minor((const double*)q, NULL, V, s, rows_Vs, cols, true , 1e-6);
        }
        else {
            q_is_minor = minor((const double*)q, NULL, V, s, rows_Vs, cols, true , 1e-6);
        }
    }

    if (!q_is_minor) { // if q is not minor, then go back to previous recursion!
        return false;
    }
    else if (q_not_in_psi && U_isnot_full && h_isnotempty) { // We keep projecting q

        // INDEPENDANT: compute W_prime
        double* W_prime = (double*)malloc(rows_Wz * cols * sizeof(double));
        if (u != NULL) { //TODO: WHAT IF u IS NULL???
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute < w , u > for u = U[-1]
                double scal_vu = scalar(W + rowVp, u, cols);

                // compute w - < w , u > . u for u = U[-1]
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp] - scal_vu * u[k];
                }
            }
        }
        else {
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute w_prime = w
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp];
                }
            }
        }

        // norm of W_prime
        double* W_prime_norm = (double*)norm_V((const double*)W_prime, rows_Wz, cols);

        // indices of independant vectors v from U
        int num_independant = 0;
        int* idx_independant = (int*)malloc(rows_Wz * sizeof(int));
        for (int j = 0; j < rows_Wz; j++) {
            if (W_prime_norm[j] > res) {
                idx_independant[num_independant++] = j;
            }
        }

        // if no independant component, we cannot go further: return q and false
        if (num_independant == 0) {

            free(W_prime);
            free(W_prime_norm);
            free(idx_independant);

            return false;
        }

        // POSITIVE: compute < q - X, W > on indices idx_independant
        double* proj_qWz = scalar_qVs_I(W, z, (const double*)q, (const int*)idx_independant, num_independant, cols);

        // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
        int num_posi_inde = 0;
        int* idx_is_posi__ = (int*)malloc(num_independant * sizeof(int));
        int* idx_is_nega__ = (int*)malloc(num_independant * sizeof(int));
        for (int j = 0; j < num_independant; j++) {
            if (proj_qWz[j] > res) {
                idx_is_posi__[num_posi_inde] = j;
                num_posi_inde++;
            }
            else {
                idx_is_nega__[j - num_posi_inde] = j;
            }
        }

        // if no positive AND independant component, we choose to not go further: return q and false
        if (num_posi_inde == 0) {

            free(W_prime);
            free(W_prime_norm);
            free(idx_independant);

            free(proj_qWz);
            free(idx_is_posi__);
            free(idx_is_nega__);

            return false;
        }

        // save a copy of the current q in new vector q_p
        double* q_p = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            q_p[j] = q[j];
        }

        // initialize state to false
        bool state = false;

        // !!!!! HERE for 1D or NOT !!!!!
        if (cols - depth == 1) { // remaining_projection_dimension is == 1

            free(idx_is_nega__);

            // normed V_prime on num_posi_inde - ON POSITIVE COMPONENTS ONLY! (unnecessary to do it on negative ones!)
            double* W_prime_normed = (double*)malloc(num_posi_inde * cols * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_independant[idx_is_posi__[j]];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_posi_inde * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_is_posi__[j];
                distances[j] = proj_qWz[k] / scalar(W + idx_independant[k] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(idx_is_posi__);
            free(proj_qWz);

            // compute argmax of distances
            int arg_max = argmax((const double*)distances, num_posi_inde);

            // update q (projected on furthest hyperplane)
            double dis_i = distances[arg_max];
            double* vec_i = W_prime_normed + arg_max * cols;
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j] - dis_i * vec_i[j];
            }
            free(W_prime_normed);
            free(distances);

            // call function project_q (go deeper in projections)
            state = project_q_2dr1(
                q, cols, 
                V, s, rows_Vs, 
                NULL, NULL, 0, 
                NULL, depth + 1, 
                res, nb_recursions
            ); // q is updated

        }
        else {

            // normed V_prime on num_independant
            double* W_prime_normed = (double*)malloc(num_independant * cols * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                int k = idx_independant[j];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                distances[j] = proj_qWz[j] / scalar(W + idx_independant[j] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(proj_qWz);

            // Sort distances! (on idx_is_posi__) - local indices for idx_independant
            int* idx_order = (int*)argsort_desc_I((const double*)distances, idx_is_posi__, num_posi_inde);
            free(idx_is_posi__);

            // compute new vectors NEW_W and new points NEW_X
            double* NEW_W = (double*)malloc(num_independant * cols * sizeof(double));
            double* NEW_z = (double*)malloc(num_independant * sizeof(double));
            double* NEW_distances = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant - num_posi_inde; j++) { // first, negative
                int idz = idx_is_nega__[j];
                int rowJj = j * cols;
                int rowJn = idz * cols;
                NEW_z[j] = scalar((const double*)q, (const double*)W_prime_normed + rowJn, cols) - distances[idz];
                NEW_distances[j] = distances[idz];
                for (int k = 0; k < cols; k++) {
                    NEW_W[rowJj + k] = W_prime_normed[rowJn + k];
                }
            }
            free(idx_is_nega__);
            for (int j = 0; j < num_posi_inde; j++) { // then, positive, sorted (ascending)
                int l = num_independant - j - 1;
                int idz = idx_order[j];
                int rowJl = l * cols;
                int rowJn = idz * cols;
                NEW_z[l] = scalar((const double*)q, (const double*)W_prime_normed + rowJn, cols) - distances[idz];
                NEW_distances[l] = distances[idz];
                for (int k = 0; k < cols; k++) {
                    NEW_W[rowJl + k] = W_prime_normed[rowJn + k];
                }
            }
            free(W_prime_normed);
            free(idx_order);
            free(distances);
            
            // Project on new hyperplane
            int i = 0;
            while (!state && i < num_posi_inde) {

                // numeral of projection
                int num_proj = num_independant - i - 1;

                // direction of projection is the vector of the furthest hyperplane
                double* NEW_u = NEW_W + num_proj * cols;

                // update q (projected on furthest hyperplane)
                double dis_i = NEW_distances[num_proj];
                for (int j = 0; j < cols; j++) {
                    q[j] = q_p[j] - dis_i * NEW_u[j];
                }

                // call function project_q (go deeper in projections)
                state = project_q_2dr1(
                    q, cols, 
                    V, s, rows_Vs, 
                    (const double*)NEW_W, (const double*)NEW_z, num_proj, 
                    (const double*)NEW_u, depth + 1, 
                    res, nb_recursions
                ); // q is dynamically updated

                i++;
            }

            free(NEW_W);
            free(NEW_z);
            free(NEW_distances);
        }

        if (state) {
            free(q_p);
            return true;
        }
        else {
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j];
            }
            free(q_p);
            return false;
        }
    }
    else if (q_not_in_psi) { // q is not in Psi AND we cannot project deeper!
        return false;
    }
    else { // q is in Psi AND is minor
        return true;
    }
}


double* algo_2dr1(const double* V, const double* s, int rows, int cols, const double* p, const double res) {

    double* proj_qVs = scalar_qVs(V, s, p, rows, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows);
    free(proj_qVs);

    // If point p is in polyhedron, then return copy of p
    if (max_proj_qVs <= res) {
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = p[k];
        }
        return q;
    }
    else {
        // initialize q
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = 0.0;
        }

        // Substract <p,v> to s
        double* s_p = (double*)malloc(rows * sizeof(double));
        for (int k = 0; k < rows; k++) {
            s_p[k] = s[k] - scalar(p, V + k * cols, cols);
        }

        // initialize depth
        int depth = 0;
        int nb_recursions = -1;

        // main algo to project iteratively q on polyhedron
        bool state = project_q_2dr1(
            q, cols, 
            (const double*)V, (const double*)s_p, rows, 
            (const double*)V, (const double*)s_p, rows, 
            NULL, depth, 
            res, &nb_recursions
        ); // q is dynamically updated

        //printf("%i ",nb_recursions);

        // free s_p
        free(s_p);

        // add p back to q
        for (int k = 0; k < cols; k++) {
            q[k] = (double)nb_recursions;// += p[k];
        }

        // If no minimum-norm point found, print warning message!
        if (!state){
            printf("WARNING: no minimum-norm point found!\n");
        }

        return q;
    }
}






///////////////////// Third VERSION rmd: SIMPLIEST, BUT TAKING ARGMAX ONLY WHEN REMAINING DIMENSION IS 1 /////////////////////
// #######################################################################################################################

// f0 recursive function
bool project_q_2dr1_rmd(
    double* q, int cols, 
    const double* V, const double* s, int rows_Vs, 
    const double* W, const double* z, int rows_Wz, 
    const double* u, int depth, 
    const double res, int *nb_recursions
) {
    *nb_recursions += 1;

    // Compute the whole < q , V > - s and its maximum
    double* proj_qVs = scalar_qVs(V, s, (const double*)q, rows_Vs, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows_Vs);
    free(proj_qVs);

    // Three conditions to keep projecting q
    bool q_not_in_psi = max_proj_qVs > res;
    bool U_isnot_full = depth < cols;
    bool h_isnotempty = rows_Wz > 0;

    // We keep projecting q
    if (q_not_in_psi && U_isnot_full && h_isnotempty) {

        // INDEPENDANT: compute W_prime
        double* W_prime = (double*)malloc(rows_Wz * cols * sizeof(double));
        if (u != NULL) { //TODO: WHAT IF u IS NULL???
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute < w , u > for u = U[-1]
                double scal_vu = scalar(W + rowVp, u, cols);

                // compute w - < w , u > . u for u = U[-1]
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp] - scal_vu * u[k];
                }
            }
        }
        else {
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute w_prime = w
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp];
                }
            }
        }

        // norm of W_prime
        double* W_prime_norm = (double*)norm_V((const double*)W_prime, rows_Wz, cols);

        // indices of independant vectors v from U
        int num_independant = 0;
        int* idx_independant = (int*)malloc(rows_Wz * sizeof(int));
        for (int j = 0; j < rows_Wz; j++) {
            if (W_prime_norm[j] > res) {
                idx_independant[num_independant++] = j;
            }
        }

        // if no independant component, we cannot go further: return q and false
        if (num_independant == 0) {

            free(W_prime);
            free(W_prime_norm);
            free(idx_independant);

            return false;
        }

        // POSITIVE: compute < q - X, W > on indices idx_independant
        double* proj_qWz = scalar_qVs_I(W, z, (const double*)q, (const int*)idx_independant, num_independant, cols);

        // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
        int num_posi_inde = 0;
        int* idx_is_posi__ = (int*)malloc(num_independant * sizeof(int));
        int* idx_is_nega__ = (int*)malloc(num_independant * sizeof(int));
        for (int j = 0; j < num_independant; j++) {
            if (proj_qWz[j] > res) {
                idx_is_posi__[num_posi_inde] = j;
                num_posi_inde++;
            }
            else {
                idx_is_nega__[j - num_posi_inde] = j;
            }
        }

        // if no positive AND independant component, we choose to not go further: return q and false
        if (num_posi_inde == 0) {

            free(W_prime);
            free(W_prime_norm);
            free(idx_independant);

            free(proj_qWz);
            free(idx_is_posi__);
            free(idx_is_nega__);

            return false;
        }

        // save a copy of the current q in new vector q_p
        double* q_p = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            q_p[j] = q[j];
        }

        // initialize state to false
        bool state = false;

        // !!!!! HERE for 1D or NOT !!!!!
        if (cols - depth == 1) { // remaining_projection_dimension is == 1

            free(idx_is_nega__);

            // normed V_prime on num_posi_inde - ON POSITIVE COMPONENTS ONLY! (unnecessary to do it on negative ones!)
            double* W_prime_normed = (double*)malloc(num_posi_inde * cols * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_independant[idx_is_posi__[j]];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_posi_inde * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_is_posi__[j];
                distances[j] = proj_qWz[k] / scalar(W + idx_independant[k] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(idx_is_posi__);
            free(proj_qWz);

            // compute argmax of distances
            int arg_max = argmax((const double*)distances, num_posi_inde);

            // update q (projected on furthest hyperplane)
            double dis_i = distances[arg_max];
            double* vec_i = W_prime_normed + arg_max * cols;
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j] - dis_i * vec_i[j];
            }
            free(W_prime_normed);
            free(distances);

            // call function project_q (go deeper in projections)
            state = project_q_2dr1_rmd(
                q, cols, 
                V, s, rows_Vs, 
                NULL, NULL, 0, 
                NULL, depth + 1, 
                res, nb_recursions
            ); // q is updated

        }
        else {

            // normed V_prime on num_independant
            double* W_prime_normed = (double*)malloc(num_independant * cols * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                int k = idx_independant[j];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                distances[j] = proj_qWz[j] / scalar(W + idx_independant[j] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(proj_qWz);

            // compute new vectors NEW_W and new points NEW_X
            double* NEW_z = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant; j++) { // first, negative
                int idz = idx_independant[j];
                NEW_z[j] = scalar((const double*)q, (const double*)W_prime_normed + idz * cols, cols) - distances[idz];
            }
            free(idx_independant);
            free(idx_is_nega__);
            free(idx_is_posi__);
            
            // Project on new hyperplane
            int i = 0;
            while (!state && i < num_independant) {

                // numeral of projection
                int num_proj = num_independant - i - 1;

                // direction of projection is the vector of the furthest hyperplane
                double* NEW_u = W_prime_normed + num_proj * cols;

                // update q (projected on furthest hyperplane)
                double dis_i = distances[num_proj];
                for (int j = 0; j < cols; j++) {
                    q[j] = q_p[j] - dis_i * NEW_u[j];
                }

                // call function project_q (go deeper in projections)
                state = project_q_2dr1_rmd(
                    q, cols, 
                    V, s, rows_Vs, 
                    (const double*)W_prime_normed, (const double*)NEW_z, num_proj, 
                    (const double*)NEW_u, depth + 1, 
                    res, nb_recursions
                ); // q is dynamically updated

                i++;
            }

            free(W_prime_normed);
            free(distances);
            free(NEW_z);
        }

        if (state) {
            free(q_p);
            return true;
        }
        else {
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j];
            }
            free(q_p);
            return false;
        }
    }
    else if (q_not_in_psi) { // q is not in Psi AND we cannot project deeper!
        return false;
    }
    else if (depth <= 1) { // q is necessarily minor, because only projected on one hyperplane at max!
        return true;
    }
    else if (!minor((const double*)q, NULL, V, s, rows_Vs, cols, true, 1e-6)) { // q is in Psi but is not minor // is 1e-6 a good res???
        return false;
    }
    else { // q is in Psi AND is minor
        return true;
    }
}


double* algo_2dr1_rmd(const double* V, const double* s, int rows, int cols, const double* p, const double res) {

    double* proj_qVs = scalar_qVs(V, s, p, rows, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows);
    free(proj_qVs);

    // If point p is in polyhedron, then return copy of p
    if (max_proj_qVs <= res) {
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = p[k];
        }
        return q;
    }
    else {
        // initialize q
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = 0.0;
        }

        // Substract <p,v> to s
        double* s_p = (double*)malloc(rows * sizeof(double));
        for (int k = 0; k < rows; k++) {
            s_p[k] = s[k] - scalar(p, V + k * cols, cols);
        }

        // Sort couples regarding s_p
        int* idx_order = (int*)argsort_desc((const double*)s_p, rows);
        double* s_ord = (double*)malloc(rows * sizeof(double));
        double* V_ord = (double*)malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            int idm = idx_order[rows - i - 1];
            s_ord[i] = s_p[idm];
            double* V_ord_i = V_ord + i * cols;
            const double* V_i = V + idm * cols;
            for (int j = 0; j < cols; j++) {
                V_ord_i[j] = V_i[j];
            }
        }
        free(idx_order);
        free(s_p);

        // initialize depth
        int depth = 0;
        int nb_recursions = -1;

        // main algo to project iteratively q on polyhedron
        bool state = project_q_2dr1_rmd(
            q, cols, 
            (const double*)V_ord, (const double*)s_ord, rows, 
            (const double*)V_ord, (const double*)s_ord, rows, 
            NULL, depth, 
            res, &nb_recursions
        ); // q is dynamically updated

        //printf("%i ",nb_recursions);

        // free s_p
        free(V_ord);
        free(s_ord);

        // add p back to q
        for (int k = 0; k < cols; k++) {
            q[k] = (double)nb_recursions;// += p[k];
        }

        // If no minimum-norm point found, print warning message!
        if (!state){
            printf("WARNING: no minimum-norm point found!\n");
        }

        return q;
    }
}








///////////////////////////// FIRST AND HALF VERSION : WITH MINIMUM H-DESCRIPTION ONLY AT depth==2 //////////////////////////////
// #######################################################################################################################








// f1 recursive function
bool project_q_05r1(
    double* q, int cols, 
    const double* V, const double* s, int rows_Vs, 
    const double* W, const double* z, int rows_Wz, 
    const double* u, int depth, 
    const double res, int *nb_recursions
) {
    *nb_recursions += 1;

    // Compute the whole < q , V > - s and its maximum
    double* proj_qVs = scalar_qVs(V, s, (const double*)q, rows_Vs, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows_Vs);
    free(proj_qVs);

    // Three conditions to keep projecting q
    bool q_not_in_psi = max_proj_qVs > res;
    bool U_isnot_full = depth < cols;
    bool h_isnotempty = rows_Wz > 0;

    // We keep projecting q
    if (q_not_in_psi && U_isnot_full && h_isnotempty) {

        // INDEPENDANT: compute W_prime
        double* W_prime = (double*)malloc(rows_Wz * cols * sizeof(double));
        if (u != NULL) { //TODO: WHAT IF u IS NULL???
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute < w , u > for u = U[-1]
                double scal_vu = scalar(W + rowVp, u, cols);

                // compute w - < w , u > . u for u = U[-1]
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp] - scal_vu * u[k];
                }
            }
        }
        else {
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute w_prime = w
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp];
                }
            }
        }

        // norm of W_prime
        double* W_prime_norm = (double*)norm_V((const double*)W_prime, rows_Wz, cols);
        
        // indices of independant vectors v from U
        int num_independant = 0;
        int* idx_independant = (int*)malloc(rows_Wz * sizeof(int));
        for (int j = 0; j < rows_Wz; j++) {
            if (W_prime_norm[j] > res) {
                idx_independant[num_independant++] = j;
            }
        }

        // if no independant component, we cannot go further: return q and false
        if (num_independant == 0) {

            free(W_prime);
            free(W_prime_norm);
            free(idx_independant);

            return false;
        }

        // save a copy of the current q in new vector q_p
        double* q_p = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            q_p[j] = q[j];
        }

        // initialize state to false
        bool state = false;

        // !!!!! HERE for 1D or NOT !!!!!
        if (cols - depth <= 1) { // remaining_projection_dimension is == 1

            // POSITIVE: compute < q - X, W > on indices idx_independant
            double* proj_qWz = scalar_qVs_I(W, z, (const double*)q, (const int*)idx_independant, num_independant, cols);

            // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
            int num_posi_inde = 0;
            int* idx_is_posi__ = (int*)malloc(num_independant * sizeof(int));
            for (int j = 0; j < num_independant; j++) {
                if (proj_qWz[j] > res) {
                    idx_is_posi__[num_posi_inde] = j;
                    num_posi_inde++;
                }
            }

            // if no positive AND independant component, we choose to not go further: return q and false
            if (num_posi_inde == 0) {

                free(W_prime);
                free(W_prime_norm);
                free(idx_independant);

                free(proj_qWz);
                free(idx_is_posi__);

                free(q_p);

                return false;
            }

            // normed V_prime on num_posi_inde - ON POSITIVE COMPONENTS ONLY! (unnecessary to do it on negative ones!)
            double* W_prime_normed = (double*)malloc(num_posi_inde * cols * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_independant[idx_is_posi__[j]];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_posi_inde * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_is_posi__[j];
                distances[j] = proj_qWz[k] / scalar(W + idx_independant[k] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(idx_is_posi__);
            free(proj_qWz);

            // compute argmax of distances
            int arg_max = argmax((const double*)distances, num_posi_inde);

            // update q (projected on furthest hyperplane)
            double dis_i = distances[arg_max];
            double* vec_i = W_prime_normed + arg_max * cols;
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j] - dis_i * vec_i[j];
            }
            free(W_prime_normed);
            free(distances);
            
            // call function project_q (go deeper in projections)
            state = project_q_05r1(
                q, cols, 
                V, s, rows_Vs, 
                NULL, NULL, 0, 
                NULL, depth + 1, 
                res, nb_recursions
            ); // q is updated

        }
        else {

            // compute < q - X, W > on indices idx_independant
            double* proj_qWz = scalar_qVs_I(W, z, (const double*)q, (const int*)idx_independant, num_independant, cols);

            bool there_is_positive = false;
            for (int j = 0; j < num_independant; j++) {
                if (proj_qWz[j] > res) {
                    there_is_positive = true;
                    break;
                }
            }
            
            // if no positive AND independant component, we choose to not go further: return q and false
            if (!there_is_positive) {

                free(W_prime);
                free(W_prime_norm);
                free(idx_independant);

                free(proj_qWz);

                free(q_p);

                return false;
            }

            // normed V_prime on num_independant
            double* W_prime_normed = (double*)malloc(num_independant * cols * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                int k = idx_independant[j];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                distances[j] = proj_qWz[j] / scalar(W + idx_independant[j] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(proj_qWz);

            // !!!!! HERE for 2D or NOT !!!!!
            if (cols - depth <= 2) { // remaining_projection_dimension is == 2

                // ########## get indices of necessary halfspaces only ##########

                double* z_prime = (double*)malloc(num_independant * sizeof(double));
                for (int j = 0; j < num_independant; j++) {
                    z_prime[j] = scalar((const double*)q, (const double*)W_prime_normed + j * cols, cols) - distances[j];
                }

                int new_num_independant; // it is theoretically assured by the minimum-H description (in previous recursion) that new_num_independant > 0
                int* new_idx_independant = keep_only_necessary_couples_idx((const double*)W_prime_normed, (const double*)z_prime, num_independant, cols, 1e-6, &new_num_independant);

                // ##############################################################

                int new_num_independant_m1 = new_num_independant - 1;

                // compute argmax of distances on new_idx_independant
                int arg_max = argmax_I((const double*)distances, (const int*)new_idx_independant, new_num_independant);

                // compute new ordered vectors NEW_W and new ordered scalars NEW_z
                double* NEW_W = (double*)malloc(new_num_independant_m1 * cols * sizeof(double));
                double* NEW_z = (double*)malloc(new_num_independant_m1 * sizeof(double));
                int current_j = 0;
                for (int j = 0; j < new_num_independant; j++) {
                    int l = new_idx_independant[j];
                    if (l != arg_max) {
                        int rowJj = current_j * cols;
                        int rowJn = l * cols;
                        NEW_z[current_j] = z_prime[l];
                        for (int k = 0; k < cols; k++) {
                            NEW_W[rowJj + k] = W_prime_normed[rowJn + k];
                        }
                        current_j++;
                    }
                }
                free(new_idx_independant);
                free(z_prime);

                double* NEW_u = W_prime_normed + arg_max * cols;
                
                double dis_i = distances[arg_max];
                for (int j = 0; j < cols; j++) {
                    q[j] = q_p[j] - dis_i * NEW_u[j];
                }
                free(distances);

                // call function project_q (go deeper in projections)
                state = project_q_05r1(
                    q, cols, 
                    V, s, rows_Vs, 
                    (const double*)NEW_W, (const double*)NEW_z, new_num_independant_m1, 
                    (const double*)NEW_u, depth + 1, 
                    res, nb_recursions
                ); // q is dynamically updated

                free(W_prime_normed);
                free(NEW_W);
                free(NEW_z);
            }
            else {

                // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
                int num_posi_inde = 0;
                int* idx_is_posi__ = (int*)malloc(num_independant * sizeof(int));
                int* idx_is_nega__ = (int*)malloc(num_independant * sizeof(int));
                for (int j = 0; j < num_independant; j++) {
                    if (distances[j] > res) {
                        idx_is_posi__[num_posi_inde] = j;
                        num_posi_inde++;
                    }
                    else {
                        idx_is_nega__[j - num_posi_inde] = j;
                    }
                }

                // Sort distances! (on idx_is_posi__) - local indices for idx_independant
                int* idx_order = (int*)argsort_desc_I((const double*)distances, (const int*)idx_is_posi__, num_posi_inde);
                free(idx_is_posi__);

                // compute new ordered vectors NEW_W and new ordered scalars NEW_z
                double* NEW_W = (double*)malloc(num_independant * cols * sizeof(double));
                double* NEW_z = (double*)malloc(num_independant * sizeof(double));
                double* NEW_distances = (double*)malloc(num_independant * sizeof(double));
                for (int j = 0; j < num_independant - num_posi_inde; j++) { // first, negative
                    int idz = idx_is_nega__[j];
                    int rowJj = j * cols;
                    int rowJn = idz * cols;
                    NEW_z[j] = scalar((const double*)q, (const double*)W_prime_normed + rowJn, cols) - distances[idz];
                    NEW_distances[j] = distances[idz];
                    for (int k = 0; k < cols; k++) {
                        NEW_W[rowJj + k] = W_prime_normed[rowJn + k];
                    }
                }
                free(idx_is_nega__);
                for (int j = 0; j < num_posi_inde; j++) { // then, positive, sorted (ascending)
                    int l = num_independant - j - 1;
                    int idz = idx_order[j];
                    int rowJl = l * cols;
                    int rowJn = idz * cols;
                    NEW_z[l] =  scalar((const double*)q, (const double*)W_prime_normed + rowJn, cols) - distances[idz];
                    NEW_distances[l] = distances[idz];
                    for (int k = 0; k < cols; k++) {
                        NEW_W[rowJl + k] = W_prime_normed[rowJn + k];
                    }
                }
                free(idx_order);
                free(W_prime_normed);
                free(distances);
                
                // Project on new hyperplane
                int i = 0;
                while (!state && i < num_posi_inde) {

                    // numeral of projection
                    int num_proj = num_independant - i - 1;

                    // direction of projection is the vector of the furthest hyperplane
                    double* NEW_u = NEW_W + num_proj * cols;

                    // update q (projected on furthest hyperplane)
                    double dis_i = NEW_distances[num_proj];
                    for (int j = 0; j < cols; j++) {
                        q[j] = q_p[j] - dis_i * NEW_u[j];
                    }

                    // call function project_q (go deeper in projections)
                    state = project_q_05r1(
                        q, cols, 
                        V, s, rows_Vs, 
                        (const double*)NEW_W, (const double*)NEW_z, num_proj, 
                        (const double*)NEW_u, depth + 1, 
                        res, nb_recursions
                    ); // q is dynamically updated

                    i++;
                }

                free(NEW_W);
                free(NEW_z);
                free(NEW_distances);
            }
        }

        if (state) {
            free(q_p);
            return true;
        }
        else {
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j];
            }
            free(q_p);
            return false;
        }
    }
    else if (q_not_in_psi) { // q is not in Psi AND we cannot project deeper!
        return false;
    }
    else if (depth <= 1) { // q is necessarily minor, because only projected on one hyperplane at max!
        return true;
    }
    else if (!minor((const double*)q, NULL, V, s, rows_Vs, cols, true, 1e-6)) { // q is in Psi but is not minor // is 1e-6 a good res???
        return false;
    }
    else { // q is in Psi AND is minor
        return true;
    }
}


double* algo_05r1(const double* V, const double* s, int rows, int cols, const double* p, const double res) {

    double* proj_qVs = scalar_qVs(V, s, p, rows, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows);
    free(proj_qVs);

    // If point p is in polyhedron, then return copy of p
    if (max_proj_qVs <= res) {
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = p[k];
        }
        return q;
    }
    else {
        // initialize q
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = 0.0;
        }

        // Substract <p,v> to s
        double* s_p = (double*)malloc(rows * sizeof(double));
        for (int k = 0; k < rows; k++) {
            s_p[k] = s[k] - scalar(p, V + k * cols, cols);
        }

        // initialize depth
        int depth = 0;
        int nb_recursions = -1;

        // main algo to project iteratively q on polyhedron
        bool state = project_q_05r1(
            q, cols, 
            (const double*)V, (const double*)s_p, rows, 
            (const double*)V, (const double*)s_p, rows, 
            NULL, depth, 
            res, &nb_recursions
        ); // q is dynamically updated

        //printf("%i ",nb_recursions);

        // free s_p
        free(s_p);

        // add p back to q
        for (int k = 0; k < cols; k++) {
            q[k] = (double)nb_recursions;// += p[k];
        }

        // If no minimum-norm point found, print warning message!
        if (!state){
            printf("WARNING: no minimum-norm point found!\n");
        }

        return q;
    }
}









///////////////////////////// FIRST AND HALF VERSION BIS : WITH MINIMUM H-DESCRIPTION AT depth==2 and at initialization //////////////////////////////
// #######################################################################################################################








//!!! recursive function is the same as project_q_05r2 !!!

double* algo_05r2(const double* V, const double* s, int rows, int cols, const double* p, const double res) {

    double* proj_qVs = scalar_qVs(V, s, p, rows, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows);
    free(proj_qVs);

    // If point p is in polyhedron, then return copy of p
    if (max_proj_qVs <= res) {
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = p[k];
        }
        return q;
    }
    else {
        // initialize q
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = 0.0;
        }

        // Copy V (it is modified in keep_only_necessary_couples_ModifyMatrices)
        double* V_p = (double*)malloc(rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                V_p[i * cols + j] = V[i * cols + j];
            }
        }

        // Substract <p,v> to s
        double* s_p = (double*)malloc(rows * sizeof(double));
        for (int k = 0; k < rows; k++) {
            s_p[k] = s[k] - scalar(p, V + k * cols, cols);
        }

        int rows_p = keep_only_necessary_couples_ModifyMatrices(V_p, s_p, rows, cols, 1e-6);

        // initialize depth
        int depth = 0;
        int nb_recursions = -1;

        // main algo to project iteratively q on polyhedron
        bool state = project_q_05r1(
            q, cols, 
            (const double*)V_p, (const double*)s_p, rows_p, 
            (const double*)V_p, (const double*)s_p, rows_p, 
            NULL, depth, 
            res, &nb_recursions
        ); // q is dynamically updated

        //printf("%i ",nb_recursions);

        // free V_p and s_p
        free(V_p);
        free(s_p);

        // add p back to q
        for (int k = 0; k < cols; k++) {
            q[k] = (double)nb_recursions;// += p[k];
        }

        // If no minimum-norm point found, print warning message!
        if (!state){
            printf("WARNING: no minimum-norm point found!\n");
        }

        return q;
    }
}








///////////////////////////// SECOND VERSION BIS : WITH MINIMUM H-DESCRIPTION AT EVERY RECURSION and considering depth==1 //////////////////////////////
// #######################################################################################################################








// f1 recursive function
bool project_q_1dr1(
    double* q, int cols, 
    const double* V, const double* s, int rows_Vs, 
    const double* W, const double* z, int rows_Wz, 
    const double* u, int depth, 
    const double res, int *nb_recursions
) {
    *nb_recursions += 1;

    // Compute the whole < q , V > - s and its maximum
    double* proj_qVs = scalar_qVs(V, s, (const double*)q, rows_Vs, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows_Vs);
    free(proj_qVs);

    // Three conditions to keep projecting q
    bool q_not_in_psi = max_proj_qVs > res;
    bool U_isnot_full = depth < cols;
    bool h_isnotempty = rows_Wz > 0;

    // We keep projecting q
    if (q_not_in_psi && U_isnot_full && h_isnotempty) {

        // INDEPENDANT: compute W_prime
        double* W_prime = (double*)malloc(rows_Wz * cols * sizeof(double));
        if (u != NULL) { //TODO: WHAT IF u IS NULL???
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute < w , u > for u = U[-1]
                double scal_vu = scalar(W + rowVp, u, cols);

                // compute w - < w , u > . u for u = U[-1]
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp] - scal_vu * u[k];
                }
            }
        }
        else {
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute w_prime = w
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp];
                }
            }
        }

        // norm of W_prime
        double* W_prime_norm = (double*)norm_V((const double*)W_prime, rows_Wz, cols);
        
        // indices of independant vectors v from U
        int num_independant = 0;
        int* idx_independant = (int*)malloc(rows_Wz * sizeof(int));
        for (int j = 0; j < rows_Wz; j++) {
            if (W_prime_norm[j] > res) {
                idx_independant[num_independant++] = j;
            }
        }

        // if no independant component, we cannot go further: return q and false
        if (num_independant == 0) {

            free(W_prime);
            free(W_prime_norm);
            free(idx_independant);

            return false;
        }

        // ##########################################################################################################################
        // INFORMATION: 
        // The minimum-H representation assures that: for all x in hyperplane H_i, if x is in polyhedron P' formed by the orthogonal 
        // projection of polyhedron P along hyperplane H_i (formula 12 in ICPRAM 2025 160), then x is in P. However, as in the while 
        // loop we successively definitively remove from (W,z) couples (W_j, z_j) that have already been considered in the loop, the 
        // polyhedron P'_{remaining H_j} misses these removed hyperplanes. Therefore, P'_{remaining H_j} is not the P' formed by all 
        // the (W'_j, z'_j) with W'_j non-zero. The property cited above is therefore not necessarily verified. Therefore, it is not 
        // assured that there exist independant OR positive components at this step here!
        // ##########################################################################################################################

        // save a copy of the current q in new vector q_p
        double* q_p = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            q_p[j] = q[j];
        }

        // initialize state to false
        bool state = false;

        // !!!!! HERE for 1D or NOT !!!!!
        if (cols - depth <= 1) { // remaining_projection_dimension is == 1

            // POSITIVE: compute < q - X, W > on indices idx_independant
            double* proj_qWz = scalar_qVs_I(W, z, (const double*)q, (const int*)idx_independant, num_independant, cols);

            // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
            int num_posi_inde = 0;
            int* idx_is_posi__ = (int*)malloc(num_independant * sizeof(int));
            for (int j = 0; j < num_independant; j++) {
                if (proj_qWz[j] > res) {
                    idx_is_posi__[num_posi_inde] = j;
                    num_posi_inde++;
                }
            }

            // if no positive AND independant component, we choose to not go further: return q and false
            if (num_posi_inde == 0) {

                free(W_prime);
                free(W_prime_norm);
                free(idx_independant);

                free(proj_qWz);
                free(idx_is_posi__);

                free(q_p);

                return false;
            }

            // normed V_prime on num_posi_inde - ON POSITIVE COMPONENTS ONLY! (unnecessary to do it on negative ones!)
            double* W_prime_normed = (double*)malloc(num_posi_inde * cols * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_independant[idx_is_posi__[j]];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_posi_inde * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_is_posi__[j];
                distances[j] = proj_qWz[k] / scalar(W + idx_independant[k] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(idx_is_posi__);
            free(proj_qWz);

            // compute argmax of distances
            int arg_max = argmax((const double*)distances, num_posi_inde);

            // update q (projected on furthest hyperplane)
            double dis_i = distances[arg_max];
            double* vec_i = W_prime_normed + arg_max * cols;
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j] - dis_i * vec_i[j];
            }
            free(W_prime_normed);
            free(distances);
            
            // call function project_q (go deeper in projections)
            state = project_q_1dr1(
                q, cols, 
                V, s, rows_Vs, 
                NULL, NULL, 0, 
                NULL, depth + 1, 
                res, nb_recursions
            ); // q is updated

        }
        else {

            // compute < q - X, W > on indices idx_independant
            double* proj_qWz = scalar_qVs_I(W, z, (const double*)q, (const int*)idx_independant, num_independant, cols);

            bool there_is_positive = false;
            for (int j = 0; j < num_independant; j++) {
                if (proj_qWz[j] > res) {
                    there_is_positive = true;
                    break;
                }
            }
            
            // if no positive AND independant component, we choose to not go further: return q and false
            if (!there_is_positive) {

                free(W_prime);
                free(W_prime_norm);
                free(idx_independant);

                free(proj_qWz);

                free(q_p);

                return false;
            }

            // normed V_prime on num_independant
            double* W_prime_normed = (double*)malloc(num_independant * cols * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                int k = idx_independant[j];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                distances[j] = proj_qWz[j] / scalar(W + idx_independant[j] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(proj_qWz);

            // ########## get indices of necessary halfspaces only ##########

            double* z_prime = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                z_prime[j] = scalar((const double*)q, (const double*)W_prime_normed + j * cols, cols) - distances[j];
            }

            int new_num_independant; // it is theoretically assured by the minimum-H description (in previous recursion) that new_num_independant > 0
            int* new_idx_independant = keep_only_necessary_couples_idx((const double*)W_prime_normed, (const double*)z_prime, num_independant, cols, 1e-6, &new_num_independant);

            // ##############################################################

            // !!!!! HERE for 2D or NOT !!!!!
            if (cols - depth <= 2) { // remaining_projection_dimension is == 2

                int new_num_independant_m1 = new_num_independant - 1;

                // compute argmax of distances on new_idx_independant
                int arg_max = argmax_I((const double*)distances, (const int*)new_idx_independant, new_num_independant);

                // compute new ordered vectors NEW_W and new ordered scalars NEW_z
                double* NEW_W = (double*)malloc(new_num_independant_m1 * cols * sizeof(double));
                double* NEW_z = (double*)malloc(new_num_independant_m1 * sizeof(double));
                int current_j = 0;
                for (int j = 0; j < new_num_independant; j++) {
                    int l = new_idx_independant[j];
                    if (l != arg_max) {
                        int rowJj = current_j * cols;
                        int rowJn = l * cols;
                        NEW_z[current_j] = z_prime[l];
                        for (int k = 0; k < cols; k++) {
                            NEW_W[rowJj + k] = W_prime_normed[rowJn + k];
                        }
                        current_j++;
                    }
                }
                free(new_idx_independant);
                free(z_prime);

                double* NEW_u = W_prime_normed + arg_max * cols;
                
                double dis_i = distances[arg_max];
                for (int j = 0; j < cols; j++) {
                    q[j] = q_p[j] - dis_i * NEW_u[j];
                }
                free(distances);

                // call function project_q (go deeper in projections)
                state = project_q_1dr1(
                    q, cols, 
                    V, s, rows_Vs, 
                    (const double*)NEW_W, (const double*)NEW_z, new_num_independant_m1, 
                    (const double*)NEW_u, depth + 1, 
                    res, nb_recursions
                ); // q is dynamically updated

                free(W_prime_normed);
                free(NEW_W);
                free(NEW_z);
            }
            else {

                // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
                int num_posi_inde = 0;
                int* idx_is_posi__ = (int*)malloc(new_num_independant * sizeof(int));
                int* idx_is_nega__ = (int*)malloc(new_num_independant * sizeof(int));
                for (int j = 0; j < new_num_independant; j++) {
                    int k = new_idx_independant[j];
                    if (distances[k] > res) {
                        idx_is_posi__[num_posi_inde] = k;
                        num_posi_inde++;
                    }
                    else {
                        idx_is_nega__[j - num_posi_inde] = k;
                    }
                }
                free(new_idx_independant);

                // Sort distances! (on idx_is_posi__) - local indices for idx_independant
                int* idx_order = (int*)argsort_desc_I((const double*)distances, (const int*)idx_is_posi__, num_posi_inde);
                free(idx_is_posi__);

                // compute new ordered vectors NEW_W and new ordered scalars NEW_z
                double* NEW_W = (double*)malloc(new_num_independant * cols * sizeof(double));
                double* NEW_z = (double*)malloc(new_num_independant * sizeof(double));
                double* NEW_distances = (double*)malloc(new_num_independant * sizeof(double));
                for (int j = 0; j < new_num_independant - num_posi_inde; j++) { // first, negative
                    int idz = idx_is_nega__[j];
                    int rowJj = j * cols;
                    int rowJn = idz * cols;
                    NEW_z[j] = z_prime[idz];
                    NEW_distances[j] = distances[idz];
                    for (int k = 0; k < cols; k++) {
                        NEW_W[rowJj + k] = W_prime_normed[rowJn + k];
                    }
                }
                free(idx_is_nega__);
                for (int j = 0; j < num_posi_inde; j++) { // then, positive, sorted (ascending)
                    int l = new_num_independant - j - 1;
                    int idz = idx_order[j];
                    int rowJl = l * cols;
                    int rowJn = idz * cols;
                    NEW_z[l] = z_prime[idz];
                    NEW_distances[l] = distances[idz];
                    for (int k = 0; k < cols; k++) {
                        NEW_W[rowJl + k] = W_prime_normed[rowJn + k];
                    }
                }
                free(idx_order);
                free(W_prime_normed);
                free(z_prime);
                free(distances);
                
                // Project on new hyperplane
                int i = 0;
                while (!state && i < num_posi_inde) {

                    // numeral of projection
                    int num_proj = new_num_independant - i - 1;

                    // direction of projection is the vector of the furthest hyperplane
                    double* NEW_u = NEW_W + num_proj * cols;

                    // update q (projected on furthest hyperplane)
                    double dis_i = NEW_distances[num_proj];
                    for (int j = 0; j < cols; j++) {
                        q[j] = q_p[j] - dis_i * NEW_u[j];
                    }

                    // call function project_q (go deeper in projections)
                    state = project_q_1dr1(
                        q, cols, 
                        V, s, rows_Vs, 
                        (const double*)NEW_W, (const double*)NEW_z, num_proj, 
                        (const double*)NEW_u, depth + 1, 
                        res, nb_recursions
                    ); // q is dynamically updated

                    i++;
                }

                free(NEW_W);
                free(NEW_z);
                free(NEW_distances);
            }
        }

        if (state) {
            free(q_p);
            return true;
        }
        else {
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j];
            }
            free(q_p);
            return false;
        }
    }
    else if (q_not_in_psi) { // q is not in Psi AND we cannot project deeper!
        return false;
    }
    else if (depth <= 1) { // q is necessarily minor, because only projected on one hyperplane at max!
        return true;
    }
    else if (!minor((const double*)q, NULL, V, s, rows_Vs, cols, true, 1e-6)) { // q is in Psi but is not minor // is 1e-6 a good res???
        return false;
    }
    else { // q is in Psi AND is minor
        return true;
    }
}


double* algo_1dr1(const double* V, const double* s, int rows, int cols, const double* p, const double res) {

    double* proj_qVs = scalar_qVs(V, s, p, rows, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows);
    free(proj_qVs);

    // If point p is in polyhedron, then return copy of p
    if (max_proj_qVs <= res) {
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = p[k];
        }
        return q;
    }
    else {
        // initialize q
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = 0.0;
        }

        // Substract <p,v> to s
        double* s_p = (double*)malloc(rows * sizeof(double));
        for (int k = 0; k < rows; k++) {
            s_p[k] = s[k] - scalar(p, V + k * cols, cols);
        }

        // initialize depth
        int depth = 0;
        int nb_recursions = -1;

        // main algo to project iteratively q on polyhedron
        bool state = project_q_1dr1(
            q, cols, 
            (const double*)V, (const double*)s_p, rows, 
            (const double*)V, (const double*)s_p, rows, 
            NULL, depth, 
            res, &nb_recursions
        ); // q is dynamically updated

        //printf("%i ",nb_recursions);

        // free s_p
        free(s_p);

        // add p back to q
        for (int k = 0; k < cols; k++) {
            q[k] = (double)nb_recursions;// += p[k];
        }

        // If no minimum-norm point found, print warning message!
        if (!state){
            printf("WARNING: no minimum-norm point found!\n");
        }

        return q;
    }
}








///////////////////////////// SECOND VERSION BIS BIIIIIIIIS : WITH MINIMUM H-DESCRIPTION AT EVERY RECURSION /////////////////
///////////////////////////// AND considering depth==1 AND using allowed-to-project-on indices //////////////////////////////
// #######################################################################################################################

//
// DESCRIPTION: COMPUTES THE MINIMUM H-DESCRIPTION USING THE FULL (W,s) INCLUDING THE COUPLES REMOVED IN THE WHILE LOOPS:
// apply 'couple_is_necessary' function on all indices of "allowed-to-be-projected-on" hyperplanes, regarding the 
// full polyhedron described by (W,z)!
//

// + best keep_only_necessary_couples : here, on all the couples in (W,z)

// + added a 'return false' if the argmax of the distances from the whole (W,z) is the same as the one of the distances 
// from (W,z) only on indices in 'new_I_allow' (in 1D and 2D only, as there is only one solution considered in these cases (the argmax))







// f1 recursive function
bool project_q_1dr2v2d2(
    double* q, int cols, 
    const double* V, const double* s, int rows_Vs, 
    const double* W, const double* z, int rows_Wz, 
    const int* I_allow, int I_rows, 
    const double* u, int depth, 
    const double res, int *nb_recursions
) {
    *nb_recursions += 1;

    // Compute the whole < q , V > - s and its maximum
    double* proj_qVs = scalar_qVs(V, s, (const double*)q, rows_Vs, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows_Vs);
    free(proj_qVs);

    // Three conditions to keep projecting q
    bool q_not_in_psi = max_proj_qVs > res;
    bool U_isnot_full = depth < cols;
    bool h_isnotempty = I_rows > 0;

    // We keep projecting q
    if (q_not_in_psi && U_isnot_full && h_isnotempty) {

        // INDEPENDANT: compute W_prime
        double* W_prime = (double*)malloc(rows_Wz * cols * sizeof(double));
        if (u != NULL) { //TODO: WHAT IF u IS NULL???
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute < w , u > for u = U[-1]
                double scal_vu = scalar(W + rowVp, u, cols);

                // compute w - < w , u > . u for u = U[-1]
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp] - scal_vu * u[k];
                }
            }
        }
        else {
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute w_prime = w
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp];
                }
            }
        }

        // norm of W_prime
        double* W_prime_norm = (double*)norm_V((const double*)W_prime, rows_Wz, cols);
        
        // indices of independant vectors v from U
        int num_independant = 0;
        int* idx_independant = (int*)malloc(rows_Wz * sizeof(int));
        for (int j = 0; j < rows_Wz; j++) {
            if (W_prime_norm[j] > res) {
                idx_independant[num_independant++] = j;
            }
        }

        // if no independant component, we cannot go further: return q and false
        if (num_independant == 0) {

            free(W_prime);
            free(W_prime_norm);
            free(idx_independant);

            return false;
        }

        // ##########################################################################################################################
        // INFORMATION: 
        // The minimum-H representation assures that: for all x in hyperplane H_i, if x is in polyhedron P' formed by the orthogonal 
        // projection of polyhedron P along hyperplane H_i (formula 12 in ICPRAM 2025 160), then x is in P.
        // ##########################################################################################################################

        // save a copy of the current q in new vector q_p
        double* q_p = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            q_p[j] = q[j];
        }

        // initialize state to false
        bool state = false;

        // !!!!! HERE for 1D or NOT !!!!!
        if (cols - depth <= 1) { // remaining_projection_dimension is == 1

            // Get indices of the intersection of idx_independant with I_allow
            int new_I_rows_on_Wz;
            int* new_I_allow_on_Wz = arg_intersection_a((const int*)idx_independant, num_independant, I_allow, I_rows, &new_I_rows_on_Wz);
            
            // if no independant component, we cannot go further: return q and false
            if (new_I_rows_on_Wz == 0) {

                free(W_prime);
                free(W_prime_norm);

                free(idx_independant);
                free(new_I_allow_on_Wz);

                free(q_p);

                return false;
            }

            // POSITIVE: compute < q - X, W > on indices new_I_allow_on_Wz
            double* proj_qWz = scalar_qVs_I(W, z, (const double*)q, (const int*)idx_independant, num_independant, cols);

            // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
            int num_posi_inde = 0;
            int* idx_is_posi__ = (int*)malloc(num_independant * sizeof(int));
            int num_posi_inde_allow = 0;
            int* idx_is_posi_allow__ = (int*)malloc(new_I_rows_on_Wz * sizeof(int));
            int id_on_allow = 0;
            for (int j = 0; j < num_independant; j++) {
                bool on_allow;
                if (id_on_allow == new_I_rows_on_Wz) {
                    on_allow = false;
                }
                else {
                    on_allow = j == new_I_allow_on_Wz[id_on_allow];
                    if (on_allow) {
                        id_on_allow ++;
                    }
                }
                if (proj_qWz[j] > res) {
                    if (on_allow) {
                        idx_is_posi_allow__[num_posi_inde_allow] = num_posi_inde;
                        num_posi_inde_allow++;
                    }
                    idx_is_posi__[num_posi_inde] = j;
                    num_posi_inde++;
                }
            }
            free(new_I_allow_on_Wz);

            // if no positive AND independant component, we choose to not go further: return q and false
            if (num_posi_inde_allow == 0) {

                free(W_prime);
                free(W_prime_norm);

                free(idx_independant);
                
                free(proj_qWz);
                free(idx_is_posi__);
                free(idx_is_posi_allow__);

                free(q_p);

                return false;
            }

            // normed V_prime on num_posi_inde - ON POSITIVE COMPONENTS ONLY! (unnecessary to do it on negative ones!)
            double* W_prime_normed = (double*)malloc(num_posi_inde * cols * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_independant[idx_is_posi__[j]];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_posi_inde * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_is_posi__[j];
                distances[j] = proj_qWz[k] / scalar(W + idx_independant[k] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(idx_is_posi__);
            free(proj_qWz);

            // compute argmax of distances
            int arg_max = argmax((const double*)distances, num_posi_inde);
            
            int arg_max_allow = argmax_I((const double*)distances, (const int*)idx_is_posi_allow__, num_posi_inde_allow);
            free(idx_is_posi_allow__);

            if (arg_max != arg_max_allow) {
                
                free(W_prime_normed);
                free(distances);

                free(q_p);

                return false;
            }

            // update q (projected on furthest hyperplane)
            double dis_i = distances[arg_max];
            double* vec_i = W_prime_normed + arg_max * cols;
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j] - dis_i * vec_i[j];
            }
            free(W_prime_normed);
            free(distances);

            // call function project_q (go deeper in projections)
            state = project_q_1dr2v2d2(
                q, cols, 
                V, s, rows_Vs, 
                NULL, NULL, 0, 
                NULL, 0, 
                NULL, depth + 1, 
                res, nb_recursions
            ); // q is updated

        }
        else {

            // Get indices of the intersection of idx_independant with I_allow
            int new_I_rows;
            int* new_I_allow = arg_intersection_a((const int*)idx_independant, num_independant, I_allow, I_rows, &new_I_rows);

            // if no independant component, we cannot go further: return q and false
            if (new_I_rows == 0) {

                free(W_prime);
                free(W_prime_norm);
                free(idx_independant);

                free(new_I_allow);
                free(q_p);

                return false;
            }

            // compute < q - X, W > on indices idx_independant
            double* proj_qWz = scalar_qVs_I(W, z, (const double*)q, (const int*)idx_independant, num_independant, cols);

            bool there_is_positive = false;
            for (int j = 0; j < new_I_rows; j++) {
                if (proj_qWz[new_I_allow[j]] > res) {
                    there_is_positive = true;
                    break;
                }
            }
            
            // if no positive AND independant component, we choose to not go further: return q and false
            if (!there_is_positive) {

                free(W_prime);
                free(W_prime_norm);
                free(idx_independant);

                free(proj_qWz);

                free(new_I_allow);
                free(q_p);

                return false;
            }

            // normed V_prime on num_independant
            double* W_prime_normed = (double*)malloc(num_independant * cols * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                int k = idx_independant[j];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                distances[j] = proj_qWz[j] / scalar(W + idx_independant[j] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(proj_qWz);

            // ########## get indices of necessary halfspaces only ##########
            // ##############################################################

            double* z_prime = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                z_prime[j] = scalar((const double*)q, (const double*)W_prime_normed + j * cols, cols) - distances[j];
            }



    // modify lists: W_prime_normed, z_prime, distances AND new_I_allow
    int new_num_independant = num_independant;
    int i = num_independant - 1;

    int new_new_I_rows = new_I_rows;
    int i_on_I = new_I_rows - 1;

    while (i >= 0) {
        
        bool i_is_currently_on_I;
        if (i_on_I < 0) {
            i_is_currently_on_I = false;
        }
        else {
            i_is_currently_on_I = new_I_allow[i_on_I] == i;
            if (i_is_currently_on_I) {
                i_on_I --;
            }
        }

        bool i_is_neessary = couple_is_necessary(i, W_prime_normed, z_prime, new_num_independant, cols, res);
        if (!i_is_neessary) {
            
            // Update new_V and new_s
            for (int j = i; j < new_num_independant - 1; j++) {
                int j1 = j + 1;
                z_prime[j] = z_prime[j1];
                distances[j] = distances[j1];
                double* W_j = W_prime_normed + j * cols;
                double* W_j1 = W_prime_normed + j1 * cols;
                for (int k = 0; k < cols; k++) {
                    W_j[k] = W_j1[k];
                }
            }

            // Update new_I_allow
            if (i_is_currently_on_I) {
                new_new_I_rows --;
                if (new_new_I_rows == 0) {
                    break;
                }
                for (int j = i_on_I + 1; j < new_new_I_rows; j++) {
                    new_I_allow[j] = new_I_allow[j + 1] - 1;
                }
            }
            else {
                for (int j = i_on_I + 1; j < new_new_I_rows; j++) {
                    new_I_allow[j] -= 1;
                }
            }

            new_num_independant --;
        }

        i --;
    }



// WARNING: new_Id (i.e. new_new_I_allow) must be sorted!!!

            // if no independant component, we cannot go further: return q and false
            if (new_new_I_rows == 0) {

                free(W_prime_normed);
                free(distances);
                free(z_prime);

                free(new_I_allow);
                free(q_p);

                return false;
            }

            there_is_positive = false;
            for (int j = 0; j < new_new_I_rows; j++) {
                if (distances[new_I_allow[j]] > res) {
                    there_is_positive = true;
                    break;
                }
            }
            
            // if no positive AND independant component, we choose to not go further: return q and false
            if (!there_is_positive) {

                free(W_prime_normed);
                free(distances);
                free(z_prime);

                free(new_I_allow);
                free(q_p);

                return false;
            }

            // ##############################################################
            // ##############################################################

            // !!!!! HERE for 2D or NOT !!!!!
            if (cols - depth <= 2) { // remaining_projection_dimension is == 2

                // compute argmax of distances on new_idx_independant
                int arg_max_i = argmax_I_i((const double*)distances, (const int*)new_I_allow, new_new_I_rows);
                int arg_max = new_I_allow[arg_max_i];

                int arg_max_on_all = argmax((const double*)distances, new_num_independant);

                if (arg_max_on_all != arg_max) { // YES!!! Because we computed the minimum H-description on the whole (W,z), and not only on new_I_allow indices!!!
                
                    free(W_prime_normed);
                    free(distances);
                    free(z_prime);

                    free(new_I_allow);
                    free(q_p);

                    return false;
                }

                // Copy the arg_max'th row from W_prime_normed in NEW_u
                double* NEW_u = (double*)malloc(cols * sizeof(double));
                double* W_am = W_prime_normed + arg_max * cols;
                for (int k = 0; k < cols; k++) {
                    NEW_u[k] = W_am[k];
                }

                // Update q
                double dis_i = distances[arg_max];
                for (int j = 0; j < cols; j++) {
                    q[j] = q_p[j] - dis_i * NEW_u[j];
                }
                free(distances);

                // Remove element at arg_max_i from new_I_allow
                new_new_I_rows = new_new_I_rows - 1;
                for (int j = arg_max_i; j < new_new_I_rows; j++) {
                    new_I_allow[j] = new_I_allow[j+1] - 1;
                }

                // compute new ordered vectors NEW_W and new ordered scalars NEW_z
                new_num_independant = new_num_independant - 1;
                for (int j = arg_max; j < new_num_independant; j++) {
                    int j1 = j + 1;
                    z_prime[j] = z_prime[j1];
                    double* W_j = W_prime_normed + j * cols;
                    double* W_j1 = W_prime_normed + j1 * cols;
                    for (int k = 0; k < cols; k++) {
                        W_j[k] = W_j1[k];
                    }
                }

                // call function project_q (go deeper in projections)
                state = project_q_1dr2v2d2(
                    q, cols, 
                    V, s, rows_Vs, 
                    (const double*)W_prime_normed, (const double*)z_prime, new_num_independant, 
                    (const int*)new_I_allow, new_new_I_rows, 
                    (const double*)NEW_u, depth + 1, 
                    res, nb_recursions
                ); // q is dynamically updated

                free(new_I_allow);
                free(W_prime_normed);
                free(z_prime);
                free(NEW_u);
            }
            else {
                
                // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
                int num_posi_inde = 0;
                int* idx_is_posi__ = (int*)malloc(new_new_I_rows * sizeof(int));
                int* idx_is_nega__ = (int*)malloc(new_new_I_rows * sizeof(int));
                for (int j = 0; j < new_new_I_rows; j++) {
                    int k = new_I_allow[j];
                    if (distances[k] > res) {
                        idx_is_posi__[num_posi_inde] = k;
                        num_posi_inde ++;
                    }
                    else {
                        idx_is_nega__[j - num_posi_inde] = k;
                    }
                }
                free(new_I_allow);

                // Sort distances! (on idx_is_posi__) - local indices for idx_independant
                int* idx_order = (int*)argsort_desc_I((const double*)distances, (const int*)idx_is_posi__, num_posi_inde);

                // compute new ordered vectors NEW_W and new ordered scalars NEW_z
                double* NEW_W = (double*)malloc(new_num_independant * cols * sizeof(double));
                double* NEW_z = (double*)malloc(new_num_independant * sizeof(double));
                double* NEW_distances = (double*)malloc(new_num_independant * sizeof(double));

                int num_front = 0;
                int num_back_posi = 0; // goes up to: num_posi_inde - 1
                int num_back_nega = 0;
                for (int u = 0; u < new_num_independant; u++) {
                    
                    int v;
                    if (num_back_posi < num_posi_inde) {
                        v = idx_is_posi__[num_back_posi];
                    }
                    else {
                        v = -1; // u is always positive, so u != v if v = -1
                    }
                    int w;
                    if (num_back_nega < new_new_I_rows - num_posi_inde) {
                        w = idx_is_nega__[num_back_nega];
                    }
                    else {
                        w = -1; // u is always positive, so u != w if w = -1
                    }

                    if (u == v) {
                        // at position "new_num_independant - num_back_posi - 1" in new lists, give old elements from position "idx_order[num_back_posi]"
                        int ix = new_num_independant - num_back_posi - 1;
                        int iy = idx_order[num_back_posi];
                        NEW_z[ix] = z_prime[iy];
                        NEW_distances[ix] = distances[iy];
                        double* NW_i = NEW_W + ix * cols;
                        double* Wpn_i = W_prime_normed + iy * cols;
                        for (int k = 0; k < cols; k++) {
                            NW_i[k] = Wpn_i[k];
                        }
                        num_back_posi ++;
                    }
                    else if (u == w) {
                        // at position "new_num_independant - num_posi_inde - num_back_nega - 1" in new lists, give old elements from position "w"
                        int iz = new_num_independant - num_posi_inde - num_back_nega - 1;
                        NEW_z[iz] = z_prime[w];
                        NEW_distances[iz] = distances[w];
                        double* NW_i = NEW_W + iz * cols;
                        double* Wpn_i = W_prime_normed + w * cols;
                        for (int k = 0; k < cols; k++) {
                            NW_i[k] = Wpn_i[k];
                        }
                        num_back_nega ++;
                    }
                    else {
                        // at position "num_front" in new lists, give old elements from position "u"
                        NEW_z[num_front] = z_prime[u];
                        NEW_distances[num_front] = distances[u];
                        double* NW_i = NEW_W + num_front * cols;
                        double* Wpn_i = W_prime_normed + u * cols;
                        for (int k = 0; k < cols; k++) {
                            NW_i[k] = Wpn_i[k];
                        }
                        num_front ++;
                    }
                }

                free(W_prime_normed);
                free(distances);
                free(z_prime);

                free(idx_is_posi__);
                free(idx_is_nega__);
                free(idx_order);

                if (num_front + num_back_nega + num_back_posi != new_num_independant) {
                    fprintf(stderr, "Erreur SUR LA SOMME!!!\n");

                    free(NEW_W);
                    free(NEW_z);
                    free(NEW_distances);

                    free(q_p);

                    return true;
                }

                // Create last_I_allow
                int* last_I_allow = (int*)malloc(new_new_I_rows * sizeof(int));
                for (int j = 0; j < new_new_I_rows; j++) {
                    last_I_allow[j] = new_num_independant - new_new_I_rows + j;
                }

                // Project on new hyperplane
                int i = 0;
                while (!state && i < num_posi_inde) {

                    // numeral of projection
                    int num_proj = new_num_independant - i - 1;

                    // direction of projection is the vector of the furthest hyperplane
                    double* NEW_u = NEW_W + num_proj * cols;

                    // update q (projected on furthest hyperplane)
                    double dis_i = NEW_distances[num_proj];
                    for (int j = 0; j < cols; j++) {
                        q[j] = q_p[j] - dis_i * NEW_u[j];
                    }

                    // call function project_q (go deeper in projections)
                    state = project_q_1dr2v2d2(
                        q, cols, 
                        V, s, rows_Vs, 
                        (const double*)NEW_W, (const double*)NEW_z, new_num_independant, 
                        (const int*)last_I_allow, new_new_I_rows - i - 1, 
                        (const double*)NEW_u, depth + 1, 
                        res, nb_recursions
                    ); // q is dynamically updated

                    i++;
                }

                free(last_I_allow);

                free(NEW_W);
                free(NEW_z);
                free(NEW_distances);
            }
        }

        if (state) {
            free(q_p);
            return true;
        }
        else {
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j];
            }
            free(q_p);
            return false;
        }
    }
    else if (q_not_in_psi) { // q is not in Psi AND we cannot project deeper!
        return false;
    }
    else if (depth <= 1) { // q is necessarily minor, because only projected on one hyperplane at max!
        return true;
    }
    else if (!minor((const double*)q, NULL, V, s, rows_Vs, cols, true, 1e-6)) { // q is in Psi but is not minor // is 1e-6 a good res???
        return false;
    }
    else { // q is in Psi AND is minor
        return true;
    }
}


double* algo_1dr2v2d2(const double* V, const double* s, int rows, int cols, const double* p, const double res) {

    double* proj_qVs = scalar_qVs(V, s, p, rows, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows);
    free(proj_qVs);

    // If point p is in polyhedron, then return copy of p
    if (max_proj_qVs <= res) {
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = p[k];
        }
        return q;
    }
    else {
        // initialize q
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = 0.0;
        }

        // Substract <p,v> to s
        double* s_p = (double*)malloc(rows * sizeof(double));
        for (int k = 0; k < rows; k++) {
            s_p[k] = s[k] - scalar(p, V + k * cols, cols);
        }

        // initialize allowed indices
        int* I_allow = (int*)malloc(rows * sizeof(int));
        for (int i = 0; i < rows; i++) {
            I_allow[i] = i;
        }

        // initialize depth
        int depth = 0;
        int nb_recursions = -1;

        // main algo to project iteratively q on polyhedron
        bool state = project_q_1dr2v2d2(
            q, cols, 
            (const double*)V, (const double*)s_p, rows, 
            (const double*)V, (const double*)s_p, rows, 
            (const int*)I_allow, rows,
            NULL, depth, 
            res, &nb_recursions
        ); // q is dynamically updated

        //printf("%i ",nb_recursions);

        // free lists
        free(I_allow);
        free(s_p);

        // add p back to q
        for (int k = 0; k < cols; k++) {
            q[k] = (double)nb_recursions;// += p[k];
        }

        // If no minimum-norm point found, print warning message!
        if (!state){
            printf("WARNING: no minimum-norm point found!\n");
        }

        return q;
    }
}









///////////////////////////// SECOND VERSION BIS BIIIIIIIIS : WITH MINIMUM H-DESCRIPTION AT EVERY RECURSION /////////////////
///////////////////////////// AND considering depth==1 AND using allowed-to-project-on indices //////////////////////////////
// #######################################################################################################################

//
// DESCRIPTION: COMPUTES THE MINIMUM H-DESCRIPTION USING THE FULL (W,s) INCLUDING THE COUPLES REMOVED IN THE WHILE LOOPS:
// apply 'couple_is_necessary' function on all indices of "allowed-to-be-projected-on" hyperplanes, regarding the 
// full polyhedron described by (W,z)!
//

// + best keep_only_necessary_couples : here, on the couples in (W,z) only from indices in 'new_I_allow'

// + added a 'return false' if the argmax of the distances from the whole (W,z) is the same as the one of the distances 
// from (W,z) only on indices in 'new_I_allow' (in 1D only, as there is only one solution considered in these cases (the argmax), 
// but not in 2D as the min-H description is not computed on the whole (W,z))






// f1 recursive function
bool project_q_1dr2v3d2(
    double* q, int cols, 
    const double* V, const double* s, int rows_Vs, 
    const double* W, const double* z, int rows_Wz, 
    const int* I_allow, int I_rows, 
    const double* u, int depth, 
    const double res, int *nb_recursions
) {
    *nb_recursions += 1;

    // Compute the whole < q , V > - s and its maximum
    double* proj_qVs = scalar_qVs(V, s, (const double*)q, rows_Vs, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows_Vs);
    free(proj_qVs);

    // Three conditions to keep projecting q
    bool q_not_in_psi = max_proj_qVs > res;
    bool U_isnot_full = depth < cols;
    bool h_isnotempty = I_rows > 0;

    // We keep projecting q
    if (q_not_in_psi && U_isnot_full && h_isnotempty) {

        // INDEPENDANT: compute W_prime
        double* W_prime = (double*)malloc(rows_Wz * cols * sizeof(double));
        if (u != NULL) { //TODO: WHAT IF u IS NULL???
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute < w , u > for u = U[-1]
                double scal_vu = scalar(W + rowVp, u, cols);

                // compute w - < w , u > . u for u = U[-1]
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp] - scal_vu * u[k];
                }
            }
        }
        else {
            for (int j = 0; j < rows_Wz; j++) {
                
                // row index of vector w = W[j]
                int rowVp = j * cols;

                // compute w_prime = w
                for (int k = 0; k < cols; k++) {
                    int idxVp = rowVp + k;
                    W_prime[idxVp] = W[idxVp];
                }
            }
        }

        // norm of W_prime
        double* W_prime_norm = (double*)norm_V((const double*)W_prime, rows_Wz, cols);
        
        // indices of independant vectors v from U
        int num_independant = 0;
        int* idx_independant = (int*)malloc(rows_Wz * sizeof(int));
        for (int j = 0; j < rows_Wz; j++) {
            if (W_prime_norm[j] > res) {
                idx_independant[num_independant++] = j;
            }
        }

        // if no independant component, we cannot go further: return q and false
        if (num_independant == 0) {

            free(W_prime);
            free(W_prime_norm);
            free(idx_independant);

            return false;
        }

        // ##########################################################################################################################
        // INFORMATION: 
        // The minimum-H representation assures that: for all x in hyperplane H_i, if x is in polyhedron P' formed by the orthogonal 
        // projection of polyhedron P along hyperplane H_i (formula 12 in ICPRAM 2025 160), then x is in P.
        // ##########################################################################################################################

        // save a copy of the current q in new vector q_p
        double* q_p = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            q_p[j] = q[j];
        }

        // initialize state to false
        bool state = false;

        // !!!!! HERE for 1D or NOT !!!!!
        if (cols - depth <= 1) { // remaining_projection_dimension is == 1

            // Get indices of the intersection of idx_independant with I_allow
            int new_I_rows_on_Wz;
            int* new_I_allow_on_Wz = arg_intersection_a((const int*)idx_independant, num_independant, I_allow, I_rows, &new_I_rows_on_Wz);
            
            // if no independant component, we cannot go further: return q and false
            if (new_I_rows_on_Wz == 0) {

                free(W_prime);
                free(W_prime_norm);

                free(idx_independant);
                free(new_I_allow_on_Wz);

                free(q_p);

                return false;
            }

            // POSITIVE: compute < q - X, W > on indices new_I_allow_on_Wz
            double* proj_qWz = scalar_qVs_I(W, z, (const double*)q, (const int*)idx_independant, num_independant, cols);

            // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
            int num_posi_inde = 0;
            int* idx_is_posi__ = (int*)malloc(num_independant * sizeof(int));
            int num_posi_inde_allow = 0;
            int* idx_is_posi_allow__ = (int*)malloc(new_I_rows_on_Wz * sizeof(int));
            int id_on_allow = 0;
            for (int j = 0; j < num_independant; j++) {
                bool on_allow;
                if (id_on_allow == new_I_rows_on_Wz) {
                    on_allow = false;
                }
                else {
                    on_allow = j == new_I_allow_on_Wz[id_on_allow];
                    if (on_allow) {
                        id_on_allow ++;
                    }
                }
                if (proj_qWz[j] > res) {
                    if (on_allow) {
                        idx_is_posi_allow__[num_posi_inde_allow] = num_posi_inde;
                        num_posi_inde_allow++;
                    }
                    idx_is_posi__[num_posi_inde] = j;
                    num_posi_inde++;
                }
            }
            free(new_I_allow_on_Wz);

            // if no positive AND independant component, we choose to not go further: return q and false
            if (num_posi_inde_allow == 0) {

                free(W_prime);
                free(W_prime_norm);

                free(idx_independant);
                
                free(proj_qWz);
                free(idx_is_posi__);
                free(idx_is_posi_allow__);

                free(q_p);

                return false;
            }

            // normed V_prime on num_posi_inde - ON POSITIVE COMPONENTS ONLY! (unnecessary to do it on negative ones!)
            double* W_prime_normed = (double*)malloc(num_posi_inde * cols * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_independant[idx_is_posi__[j]];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_posi_inde * sizeof(double));
            for (int j = 0; j < num_posi_inde; j++) {
                int k = idx_is_posi__[j];
                distances[j] = proj_qWz[k] / scalar(W + idx_independant[k] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(idx_is_posi__);
            free(proj_qWz);

            // compute argmax of distances
            int arg_max = argmax((const double*)distances, num_posi_inde);
            
            int arg_max_allow = argmax_I((const double*)distances, (const int*)idx_is_posi_allow__, num_posi_inde_allow);
            free(idx_is_posi_allow__);

            if (arg_max != arg_max_allow) {
                
                free(W_prime_normed);
                free(distances);

                free(q_p);

                return false;
            }

            // update q (projected on furthest hyperplane)
            double dis_i = distances[arg_max];
            double* vec_i = W_prime_normed + arg_max * cols;
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j] - dis_i * vec_i[j];
            }
            free(W_prime_normed);
            free(distances);

            // call function project_q (go deeper in projections)
            state = project_q_1dr2v3d2(
                q, cols, 
                V, s, rows_Vs, 
                NULL, NULL, 0, 
                NULL, 0, 
                NULL, depth + 1, 
                res, nb_recursions
            ); // q is updated

        }
        else {

            // Get indices of the intersection of idx_independant with I_allow
            int new_I_rows;
            int* new_I_allow = arg_intersection_a((const int*)idx_independant, num_independant, I_allow, I_rows, &new_I_rows);

            // if no independant component, we cannot go further: return q and false
            if (new_I_rows == 0) {

                free(W_prime);
                free(W_prime_norm);
                free(idx_independant);

                free(new_I_allow);
                free(q_p);

                return false;
            }

            // compute < q - X, W > on indices idx_independant
            double* proj_qWz = scalar_qVs_I(W, z, (const double*)q, (const int*)idx_independant, num_independant, cols);

            bool there_is_positive = false;
            for (int j = 0; j < new_I_rows; j++) {
                if (proj_qWz[new_I_allow[j]] > res) {
                    there_is_positive = true;
                    break;
                }
            }
            
            // if no positive AND independant component, we choose to not go further: return q and false
            if (!there_is_positive) {

                free(W_prime);
                free(W_prime_norm);
                free(idx_independant);

                free(proj_qWz);

                free(new_I_allow);
                free(q_p);

                return false;
            }

            // normed V_prime on num_independant
            double* W_prime_normed = (double*)malloc(num_independant * cols * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                int k = idx_independant[j];
                int rowVpx = j * cols;
                int rowVpy = k * cols;
                double norm_j = W_prime_norm[k];
                for (int t = 0; t < cols; t++) {
                    W_prime_normed[rowVpx + t] = W_prime[rowVpy + t] / norm_j;
                }
            }
            free(W_prime);
            free(W_prime_norm);

            // compute projection distances
            double* distances = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                distances[j] = proj_qWz[j] / scalar(W + idx_independant[j] * cols, (const double*)W_prime_normed + j * cols, cols);
            }
            free(idx_independant);
            free(proj_qWz);

            // ########## get indices of necessary halfspaces only ##########
            // ##############################################################

            double* z_prime = (double*)malloc(num_independant * sizeof(double));
            for (int j = 0; j < num_independant; j++) {
                z_prime[j] = scalar((const double*)q, (const double*)W_prime_normed + j * cols, cols) - distances[j];
            }



    // modify lists: W_prime_normed, z_prime, distances AND new_I_allow
    int new_num_independant = num_independant;
    int new_new_I_rows = new_I_rows;
    int i = new_I_rows - 1;

    while (i >= 0) {

        int id = new_I_allow[i];

        bool i_is_neessary = couple_is_necessary(id, W_prime_normed, z_prime, new_num_independant, cols, res);
        if (!i_is_neessary) {
            
            // Update new_V and new_s
            for (int j = id; j < new_num_independant - 1; j++) {
                int j1 = j + 1;
                z_prime[j] = z_prime[j1];
                distances[j] = distances[j1];
                double* W_j = W_prime_normed + j * cols;
                double* W_j1 = W_prime_normed + j1 * cols;
                for (int k = 0; k < cols; k++) {
                    W_j[k] = W_j1[k];
                }
            }

            // Update new_I_allow
            for (int j = i; j < new_new_I_rows - 1; j++) {
                new_I_allow[j] = new_I_allow[j + 1] - 1;
            }

            new_num_independant --;
            new_new_I_rows --;
        }

        i --;
    }



// WARNING: new_Id (i.e. new_new_I_allow) must be sorted!!!

            // if no independant component, we cannot go further: return q and false
            if (new_new_I_rows == 0) {

                free(W_prime_normed);
                free(distances);
                free(z_prime);

                free(new_I_allow);
                free(q_p);

                return false;
            }

            there_is_positive = false;
            for (int j = 0; j < new_new_I_rows; j++) {
                if (distances[new_I_allow[j]] > res) {
                    there_is_positive = true;
                    break;
                }
            }
            
            // if no positive AND independant component, we choose to not go further: return q and false
            if (!there_is_positive) {

                free(W_prime_normed);
                free(distances);
                free(z_prime);

                free(new_I_allow);
                free(q_p);

                return false;
            }

            // ##############################################################
            // ##############################################################

            // !!!!! HERE for 2D or NOT !!!!!
            if (cols - depth <= 2) { // remaining_projection_dimension is == 2

                // compute argmax of distances on new_idx_independant
                int arg_max_i = argmax_I_i((const double*)distances, (const int*)new_I_allow, new_new_I_rows);
                int arg_max = new_I_allow[arg_max_i];

                // Copy the arg_max'th row from W_prime_normed in NEW_u
                double* NEW_u = (double*)malloc(cols * sizeof(double));
                double* W_am = W_prime_normed + arg_max * cols;
                for (int k = 0; k < cols; k++) {
                    NEW_u[k] = W_am[k];
                }

                // Update q
                double dis_i = distances[arg_max];
                for (int j = 0; j < cols; j++) {
                    q[j] = q_p[j] - dis_i * NEW_u[j];
                }
                free(distances);

                // Remove element at arg_max_i from new_I_allow
                new_new_I_rows = new_new_I_rows - 1;
                for (int j = arg_max_i; j < new_new_I_rows; j++) {
                    new_I_allow[j] = new_I_allow[j+1] - 1;
                }

                // compute new ordered vectors NEW_W and new ordered scalars NEW_z
                new_num_independant = new_num_independant - 1;
                for (int j = arg_max; j < new_num_independant; j++) {
                    int j1 = j + 1;
                    z_prime[j] = z_prime[j1];
                    double* W_j = W_prime_normed + j * cols;
                    double* W_j1 = W_prime_normed + j1 * cols;
                    for (int k = 0; k < cols; k++) {
                        W_j[k] = W_j1[k];
                    }
                }

                // call function project_q (go deeper in projections)
                state = project_q_1dr2v3d2(
                    q, cols, 
                    V, s, rows_Vs, 
                    (const double*)W_prime_normed, (const double*)z_prime, new_num_independant, 
                    (const int*)new_I_allow, new_new_I_rows, 
                    (const double*)NEW_u, depth + 1, 
                    res, nb_recursions
                ); // q is dynamically updated

                free(new_I_allow);
                free(W_prime_normed);
                free(z_prime);
                free(NEW_u);
            }
            else {
                
                // POSITIVE AND INDEPENDANT: indices of positive AND independant vectors
                int num_posi_inde = 0;
                int* idx_is_posi__ = (int*)malloc(new_new_I_rows * sizeof(int));
                int* idx_is_nega__ = (int*)malloc(new_new_I_rows * sizeof(int));
                for (int j = 0; j < new_new_I_rows; j++) {
                    int k = new_I_allow[j];
                    if (distances[k] > res) {
                        idx_is_posi__[num_posi_inde] = k;
                        num_posi_inde ++;
                    }
                    else {
                        idx_is_nega__[j - num_posi_inde] = k;
                    }
                }
                free(new_I_allow);

                // Sort distances! (on idx_is_posi__) - local indices for idx_independant
                int* idx_order = (int*)argsort_desc_I((const double*)distances, (const int*)idx_is_posi__, num_posi_inde);

                // compute new ordered vectors NEW_W and new ordered scalars NEW_z
                double* NEW_W = (double*)malloc(new_num_independant * cols * sizeof(double));
                double* NEW_z = (double*)malloc(new_num_independant * sizeof(double));
                double* NEW_distances = (double*)malloc(new_num_independant * sizeof(double));

                int num_front = 0;
                int num_back_posi = 0; // goes up to: num_posi_inde - 1
                int num_back_nega = 0;
                for (int u = 0; u < new_num_independant; u++) {
                    
                    int v;
                    if (num_back_posi < num_posi_inde) {
                        v = idx_is_posi__[num_back_posi];
                    }
                    else {
                        v = -1; // u is always positive, so u != v if v = -1
                    }
                    int w;
                    if (num_back_nega < new_new_I_rows - num_posi_inde) {
                        w = idx_is_nega__[num_back_nega];
                    }
                    else {
                        w = -1; // u is always positive, so u != w if w = -1
                    }

                    if (u == v) {
                        // at position "new_num_independant - num_back_posi - 1" in new lists, give old elements from position "idx_order[num_back_posi]"
                        int ix = new_num_independant - num_back_posi - 1;
                        int iy = idx_order[num_back_posi];
                        NEW_z[ix] = z_prime[iy];
                        NEW_distances[ix] = distances[iy];
                        double* NW_i = NEW_W + ix * cols;
                        double* Wpn_i = W_prime_normed + iy * cols;
                        for (int k = 0; k < cols; k++) {
                            NW_i[k] = Wpn_i[k];
                        }
                        num_back_posi ++;
                    }
                    else if (u == w) {
                        // at position "new_num_independant - num_posi_inde - num_back_nega - 1" in new lists, give old elements from position "w"
                        int iz = new_num_independant - num_posi_inde - num_back_nega - 1;
                        NEW_z[iz] = z_prime[w];
                        NEW_distances[iz] = distances[w];
                        double* NW_i = NEW_W + iz * cols;
                        double* Wpn_i = W_prime_normed + w * cols;
                        for (int k = 0; k < cols; k++) {
                            NW_i[k] = Wpn_i[k];
                        }
                        num_back_nega ++;
                    }
                    else {
                        // at position "num_front" in new lists, give old elements from position "u"
                        NEW_z[num_front] = z_prime[u];
                        NEW_distances[num_front] = distances[u];
                        double* NW_i = NEW_W + num_front * cols;
                        double* Wpn_i = W_prime_normed + u * cols;
                        for (int k = 0; k < cols; k++) {
                            NW_i[k] = Wpn_i[k];
                        }
                        num_front ++;
                    }
                }

                free(W_prime_normed);
                free(distances);
                free(z_prime);

                free(idx_is_posi__);
                free(idx_is_nega__);
                free(idx_order);

                if (num_front + num_back_nega + num_back_posi != new_num_independant) {
                    fprintf(stderr, "Erreur SUR LA SOMME!!!\n");

                    free(NEW_W);
                    free(NEW_z);
                    free(NEW_distances);

                    free(q_p);

                    return true;
                }

                // Create last_I_allow
                int* last_I_allow = (int*)malloc(new_new_I_rows * sizeof(int));
                for (int j = 0; j < new_new_I_rows; j++) {
                    last_I_allow[j] = new_num_independant - new_new_I_rows + j;
                }

                // Project on new hyperplane
                int i = 0;
                while (!state && i < num_posi_inde) {

                    // numeral of projection
                    int num_proj = new_num_independant - i - 1;

                    // direction of projection is the vector of the furthest hyperplane
                    double* NEW_u = NEW_W + num_proj * cols;

                    // update q (projected on furthest hyperplane)
                    double dis_i = NEW_distances[num_proj];
                    for (int j = 0; j < cols; j++) {
                        q[j] = q_p[j] - dis_i * NEW_u[j];
                    }

                    // call function project_q (go deeper in projections)
                    state = project_q_1dr2v3d2(
                        q, cols, 
                        V, s, rows_Vs, 
                        (const double*)NEW_W, (const double*)NEW_z, new_num_independant, 
                        (const int*)last_I_allow, new_new_I_rows - i - 1, 
                        (const double*)NEW_u, depth + 1, 
                        res, nb_recursions
                    ); // q is dynamically updated

                    i++;
                }

                free(last_I_allow);

                free(NEW_W);
                free(NEW_z);
                free(NEW_distances);
            }
        }

        if (state) {
            free(q_p);
            return true;
        }
        else {
            for (int j = 0; j < cols; j++) {
                q[j] = q_p[j];
            }
            free(q_p);
            return false;
        }
    }
    else if (q_not_in_psi) { // q is not in Psi AND we cannot project deeper!
        return false;
    }
    else if (depth <= 1) { // q is necessarily minor, because only projected on one hyperplane at max!
        return true;
    }
    else if (!minor((const double*)q, NULL, V, s, rows_Vs, cols, true, 1e-6)) { // q is in Psi but is not minor // is 1e-6 a good res???
        return false;
    }
    else { // q is in Psi AND is minor
        return true;
    }
}


double* algo_1dr2v3d2(const double* V, const double* s, int rows, int cols, const double* p, const double res) {

    double* proj_qVs = scalar_qVs(V, s, p, rows, cols);
    double max_proj_qVs = max((const double*)proj_qVs, rows);
    free(proj_qVs);

    // If point p is in polyhedron, then return copy of p
    if (max_proj_qVs <= res) {
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = p[k];
        }
        return q;
    }
    else {
        // initialize q
        double* q = (double*)malloc(cols * sizeof(double));
        for (int k = 0; k < cols; k++) {
            q[k] = 0.0;
        }

        // Substract <p,v> to s
        double* s_p = (double*)malloc(rows * sizeof(double));
        for (int k = 0; k < rows; k++) {
            s_p[k] = s[k] - scalar(p, V + k * cols, cols);
        }

        // initialize allowed indices
        int* I_allow = (int*)malloc(rows * sizeof(int));
        for (int i = 0; i < rows; i++) {
            I_allow[i] = i;
        }

        // initialize depth
        int depth = 0;
        int nb_recursions = -1;

        // main algo to project iteratively q on polyhedron
        bool state = project_q_1dr2v3d2(
            q, cols, 
            (const double*)V, (const double*)s_p, rows, 
            (const double*)V, (const double*)s_p, rows, 
            (const int*)I_allow, rows,
            NULL, depth, 
            res, &nb_recursions
        ); // q is dynamically updated

        //printf("%i ",nb_recursions);

        // free lists
        free(I_allow);
        free(s_p);

        // add p back to q
        for (int k = 0; k < cols; k++) {
            q[k] = (double)nb_recursions;// += p[k];
        }

        // If no minimum-norm point found, print warning message!
        if (!state){
            printf("WARNING: no minimum-norm point found!\n");
        }

        return q;
    }
}

















//////////////////////////////////////////// FUNCTIONS FOR MULTI PYTHON USAGE ////////////////////////////////////////////
// #######################################################################################################################
// #######################################################################################################################
// #######################################################################################################################

// Fonction pour executer algo_i sur l'ensemble des points et l'ensemble des polyhedres definis par V et s
double* minimum_norm_points_to_polyhedra(
    const double* l_V, 
    const double* l_s, 
    const double* points, 
    const int* l_rows, 
    int n_polyhedra, 
    int n_points, 
    int cols, 
    double res, 
    const char* method
) {
    if (strcmp(method, "0") == 0) {
        double* qoints = (double*)malloc(n_points * n_polyhedra * cols * sizeof(double));
        for (int i = 0; i < n_points; i++) {
            
            int total_row = 0;
            
            double* points_i = (double*)points + i * cols;

            for (int j = 0; j < n_polyhedra; j++) {

                int rows_j = l_rows[j];

                double* l_V_j = (double*)l_V + total_row * cols;
                double* l_s_j = (double*)l_s + total_row;

                double* qij = (double*)algo_0dr1((const double*)l_V_j, (const double*)l_s_j, (int)rows_j, cols, (const double*)points_i, res);

                double* qoints_ij = qoints + i * n_polyhedra * cols + j * cols;
                for (int k = 0; k < cols; k++) {
                    qoints_ij[k] = qij[k];
                }

                free(qij);

                total_row += rows_j;
            }
        }
        return qoints;
    }
    else if (strcmp(method, "1") == 0) {
        double* qoints = (double*)malloc(n_points * n_polyhedra * cols * sizeof(double));
        for (int i = 0; i < n_points; i++) {
            
            int total_row = 0;
            
            double* points_i = (double*)points + i * cols;

            for (int j = 0; j < n_polyhedra; j++) {

                int rows_j = l_rows[j];

                double* l_V_j = (double*)l_V + total_row * cols;
                double* l_s_j = (double*)l_s + total_row;

                double* qij = (double*)algo_1dr1((const double*)l_V_j, (const double*)l_s_j, (int)rows_j, cols, (const double*)points_i, res);

                double* qoints_ij = qoints + i * n_polyhedra * cols + j * cols;
                for (int k = 0; k < cols; k++) {
                    qoints_ij[k] = qij[k];
                }

                free(qij);

                total_row += rows_j;
            }
        }
        return qoints;
    }
    else if (strcmp(method, "2") == 0) {
        double* qoints = (double*)malloc(n_points * n_polyhedra * cols * sizeof(double));
        for (int i = 0; i < n_points; i++) {
            
            int total_row = 0;
            
            double* points_i = (double*)points + i * cols;

            for (int j = 0; j < n_polyhedra; j++) {

                int rows_j = l_rows[j];

                double* l_V_j = (double*)l_V + total_row * cols;
                double* l_s_j = (double*)l_s + total_row;

                double* qij = (double*)algo_1dr2v2d2((const double*)l_V_j, (const double*)l_s_j, (int)rows_j, cols, (const double*)points_i, res);

                double* qoints_ij = qoints + i * n_polyhedra * cols + j * cols;
                for (int k = 0; k < cols; k++) {
                    qoints_ij[k] = qij[k];
                }

                free(qij);

                total_row += rows_j;
            }
        }
        return qoints;
    }
    else if (strcmp(method, "3") == 0) {
        double* qoints = (double*)malloc(n_points * n_polyhedra * cols * sizeof(double));
        for (int i = 0; i < n_points; i++) {
            
            int total_row = 0;
            
            double* points_i = (double*)points + i * cols;

            for (int j = 0; j < n_polyhedra; j++) {

                int rows_j = l_rows[j];

                double* l_V_j = (double*)l_V + total_row * cols;
                double* l_s_j = (double*)l_s + total_row;

                double* qij = (double*)algo_1dr2v3d2((const double*)l_V_j, (const double*)l_s_j, (int)rows_j, cols, (const double*)points_i, res);

                double* qoints_ij = qoints + i * n_polyhedra * cols + j * cols;
                for (int k = 0; k < cols; k++) {
                    qoints_ij[k] = qij[k];
                }

                free(qij);

                total_row += rows_j;
            }
        }
        return qoints;
    }
    else if (strcmp(method, "4") == 0) {
        double* qoints = (double*)malloc(n_points * n_polyhedra * cols * sizeof(double));
        for (int i = 0; i < n_points; i++) {
            
            int total_row = 0;
            
            double* points_i = (double*)points + i * cols;

            for (int j = 0; j < n_polyhedra; j++) {

                int rows_j = l_rows[j];

                double* l_V_j = (double*)l_V + total_row * cols;
                double* l_s_j = (double*)l_s + total_row;

                double* qij = (double*)algo_05r1((const double*)l_V_j, (const double*)l_s_j, (int)rows_j, cols, (const double*)points_i, res);

                double* qoints_ij = qoints + i * n_polyhedra * cols + j * cols;
                for (int k = 0; k < cols; k++) {
                    qoints_ij[k] = qij[k];
                }

                free(qij);

                total_row += rows_j;
            }
        }
        return qoints;
    }
    else if (strcmp(method, "5") == 0) {
        double* qoints = (double*)malloc(n_points * n_polyhedra * cols * sizeof(double));
        for (int i = 0; i < n_points; i++) {
            
            int total_row = 0;
            
            double* points_i = (double*)points + i * cols;

            for (int j = 0; j < n_polyhedra; j++) {

                int rows_j = l_rows[j];

                double* l_V_j = (double*)l_V + total_row * cols;
                double* l_s_j = (double*)l_s + total_row;

                double* qij = (double*)algo_2dr1((const double*)l_V_j, (const double*)l_s_j, (int)rows_j, cols, (const double*)points_i, res);

                double* qoints_ij = qoints + i * n_polyhedra * cols + j * cols;
                for (int k = 0; k < cols; k++) {
                    qoints_ij[k] = qij[k];
                }

                free(qij);

                total_row += rows_j;
            }
        }
        return qoints;
    }
    else {
        fprintf(stderr, "Erreur : Le parametre 'method' doit etre dans {'0', '1', '2', '3'}\n");
        exit(EXIT_FAILURE);
    }
}


// Fonction "wrappee" pour Python
static PyObject* minimum_norm_points_to_polyhedra_py(PyObject *self, PyObject *args) {
    
    PyArrayObject *V, *s, *points, *Lrows;
    int n_polyhedra, n_points, cols;
    double res;
    const char *method;

    if (!PyArg_ParseTuple(args, "O!O!O!O!iiids", 
                          &PyArray_Type, &V, 
                          &PyArray_Type, &s, 
                          &PyArray_Type, &points, 
                          &PyArray_Type, &Lrows, 
                          &n_polyhedra, &n_points, &cols, 
                          &res, &method)) {
        return NULL; // Si les arguments ne sont pas corrects
    }

    // Verification des types et dimensions des tableaux
    if (PyArray_NDIM(V) != 1 || PyArray_TYPE(V) != NPY_DOUBLE ||
        PyArray_NDIM(s) != 1 || PyArray_TYPE(s) != NPY_DOUBLE ||
        PyArray_NDIM(points) != 1 || PyArray_TYPE(points) != NPY_DOUBLE ||
        PyArray_NDIM(Lrows) != 1 || PyArray_TYPE(Lrows) != NPY_INT) {
        PyErr_SetString(PyExc_TypeError, "Les tableaux doivent être 1D de type double ou int.");
        return NULL;
    }

    const double *l_V = (const double *) PyArray_DATA(V);
    const double *l_s = (const double *) PyArray_DATA(s);
    const double *l_points = (const double *) PyArray_DATA(points);
    const int *l_rows = (const int *) PyArray_DATA(Lrows);

    // Appel de la fonction C
    double *results = minimum_norm_points_to_polyhedra(
        l_V, l_s, l_points, l_rows, 
        n_polyhedra, n_points, cols, 
        res, method
    );

    // Creation du tableau de sortie NumPy
    npy_intp dims[1] = {n_points * n_polyhedra * cols};
    PyObject *output_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double *data_out = (double *) PyArray_DATA((PyArrayObject *) output_array);

    // Copie des resultats
    for (int i = 0; i < n_points; i++) {
        int idx = i * n_polyhedra * cols;
        for (int j = 0; j < n_polyhedra; j++) {
            int idy = idx + j * cols;
            for (int k = 0; k < cols; k++) {
                int idz = idy + k;
                data_out[idz] = results[idz];
            }
        }
    }

    // Libere la memoire allouée en C
    free(results);

    // Retourne le tableau Python (NumPy)
    return output_array;
}

// Table des methodes accessibles depuis Python
static PyMethodDef MyModuleMethods[] = {
    {"minimum_norm_points_to_polyhedra", minimum_norm_points_to_polyhedra_py, METH_VARARGS, "Execute la fonction C minimum_norm_points_to_polyhedra."},
    {NULL, NULL, 0, NULL}  // Indicateur de fin de la table
};

// Definition du module
static struct PyModuleDef min_norm_point_module_definition = {
    PyModuleDef_HEAD_INIT,
    "min_norm_point_module",   // Nom du module
    "Un module qui rend minimum_norm_points_to_polyhedra compatible avec Python.",
    -1,             // Taille de l'état du module
    MyModuleMethods // Liste des méthodes du module
};

// Fonction d'initialisation du module
PyMODINIT_FUNC PyInit_min_norm_point_module(void) {
    import_array();  // Necessaire pour initialiser NumPy dans le contexte C
    return PyModule_Create(&min_norm_point_module_definition);
}
