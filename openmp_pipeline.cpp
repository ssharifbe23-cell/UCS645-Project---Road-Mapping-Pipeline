#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;

int main() {

    int n = 10000000;
    vector<int> data(n, 100);

    // control threads
    omp_set_num_threads(4);

    double start = omp_get_wtime();

    cout << "Starting OpenMP Parallel Pipeline...\n";

    #pragma omp parallel sections
    {
        // ---------------- STAGE 1: PREPROCESS ----------------
        #pragma omp section
        {
            cout << "Preprocessing Stage Running...\n";

            #pragma omp parallel for schedule(dynamic, 1000)
            for(int i = 0; i < n; i++) {
                data[i] = data[i] * 2;

                if(i < 5)
                    printf("Preprocess Thread %d -> index %d\n", omp_get_thread_num(), i);
            }
        }

        // ---------------- STAGE 2: DETECTION ----------------
        #pragma omp section
        {
            cout << "Detection Stage Running...\n";

            #pragma omp parallel for schedule(dynamic, 1000)
            for(int i = 0; i < n; i++) {
                data[i] = data[i] + 5;

                if(i < 5)
                    printf("Detect Thread %d -> index %d\n", omp_get_thread_num(), i);
            }
        }

        // ---------------- STAGE 3: ANALYSIS ----------------
        #pragma omp section
        {
            cout << "Analysis Stage Running...\n";

            int sum = 0;

            #pragma omp parallel for reduction(+:sum) schedule(dynamic, 1000)
            for(int i = 0; i < n; i++) {
                sum += data[i];

                if(i < 5)
                    printf("Analyze Thread %d -> index %d\n", omp_get_thread_num(), i);
            }

            cout << "Final Analysis (Sum): " << sum << endl;
        }
    }

    double end = omp_get_wtime();

    cout << "\nExecution Time: " << end - start << " seconds" << endl;

    return 0;
}