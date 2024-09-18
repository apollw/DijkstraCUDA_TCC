#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_VERTICES /*16*/ 256 /*512*/ /*1024*/ /*2048*/ /*4096*/ /*8192*/ /*16384*/
#define MIN_PESO 1
#define MAX_PESO 20

#define BLOCK_SIZE 256

//Estrutura que Representa um Nó no Grafo
struct No {
	int verticeDestino;
	int pesoAresta;
	struct No* proxNo;
};

//Estrutura que Representa o Grafo
struct Grafo {
	struct No* cabeca[NUM_VERTICES];
	int numVertices;
};

//Função que Representa Novo Nó no Grafo
struct No* criarNo(int v, int p) {
	struct No* novoNo = (struct No*)malloc(sizeof(struct No));
	novoNo->verticeDestino = v;
	novoNo->pesoAresta = p;
	novoNo->proxNo = NULL;
	return novoNo;
}

//Função que Cria o Grafo do Problema
struct Grafo* criarGrafo(int vertices) {
	struct Grafo* grafo = (struct Grafo*)malloc(sizeof(struct Grafo));
	grafo->numVertices = vertices;
	for (int i = 0; i < vertices; i++) {
		grafo->cabeca[i] = NULL;
	}
	return grafo;
}

//Função de Adicionar Arestas entre os Nós do Grafo
void adicionarAresta(struct Grafo* grafo, int orig, int dest, int peso) {
	struct No* novoNo = criarNo(dest, peso);
	novoNo->proxNo = grafo->cabeca[orig];
	grafo->cabeca[orig] = novoNo;
}

//Função que Imprime o Grafo na Tela
void imprimirGrafo(struct Grafo* grafo) {
	printf("\nGrafo:\n");
	for (int i = 0; i < grafo->numVertices; i++) {
		struct No* temp = grafo->cabeca[i];
		printf("Vertice %d: ", i);
		while (temp != NULL) {
			printf("(%d,%d) -> ", temp->verticeDestino, temp->pesoAresta);
			temp = temp->proxNo;
		}
		printf("NULL\n");
	}
}

// Função para salvar o grafo em um arquivo
void salvarGrafo(struct Grafo* grafo, const char* nomeArquivo) {
	FILE* arquivo = fopen(nomeArquivo, "w");
	if (arquivo == NULL) {
		printf("Erro ao abrir o arquivo %s.\n", nomeArquivo);
		return;
	}

	// Escreve o número de vértices no arquivo
	fprintf(arquivo, "%d\n", grafo->numVertices);

	// Escreve as arestas do grafo no arquivo
	for (int i = 0; i < grafo->numVertices; i++) {
		struct No* temp = grafo->cabeca[i];
		while (temp != NULL) {
			fprintf(arquivo, "%d %d %d\n", i, temp->verticeDestino, temp->pesoAresta);
			temp = temp->proxNo;
		}
	}

	fclose(arquivo);
	printf("Grafo salvo com sucesso no arquivo %s.\n", nomeArquivo);
}

// Função para carregar o grafo de um arquivo
struct Grafo* carregarGrafo(const char* nomeArquivo) {
	FILE* arquivo = fopen(nomeArquivo, "r");
	if (arquivo == NULL) {
		printf("Erro ao abrir o arquivo %s.\n", nomeArquivo);
		return NULL;
	}

	int numVertices;
	fscanf(arquivo, "%d", &numVertices);

	struct Grafo* grafo = criarGrafo(numVertices);

	int origem, destino, peso;
	while (fscanf(arquivo, "%d %d %d", &origem, &destino, &peso) == 3) {
		adicionarAresta(grafo, origem, destino, peso);
	}

	fclose(arquivo);
	printf("Grafo carregado com sucesso do arquivo %s.\n", nomeArquivo);
	return grafo;
}

// Kernel para encontrar o vértice não visitado com a menor distância local
__global__ void findMinDistance(int* d, bool* visited, int* minDistIndex, int* minDistValue, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n && !visited[tid]) {
		int dist = d[tid];
		atomicMin(minDistValue, dist);
		if (dist == *minDistValue) {
			*minDistIndex = tid;
		}
	}
}

// Função para encontrar o próximo vértice a ser visitado
int findNextVertex(int* d_dev, bool* visited_dev, int* minDistIndex_dev, int* minDistValue_dev, int n) {
	int minDistIndex = -1;
	int minDistValue = INT_MAX;

	cudaMemcpy(minDistValue_dev, &minDistValue, sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	findMinDistance << <numBlocks, BLOCK_SIZE >> > (d_dev, visited_dev, minDistIndex_dev, minDistValue_dev, n);

	cudaMemcpy(&minDistIndex, minDistIndex_dev, sizeof(int), cudaMemcpyDeviceToHost);

	return minDistIndex;
}

// Função para executar o algoritmo de Dijkstra paralelizado usando CUDA
void dijkstra_CUDA(struct Grafo* grafo, int inicio) {
	int distancias[NUM_VERTICES];
	bool visitados[NUM_VERTICES];

	for (int i = 0; i < NUM_VERTICES; i++) {
		distancias[i] = INT_MAX;
		visitados[i] = false;
	}

	distancias[inicio] = 0;

	int* d_dev;
	bool* visited_dev;
	int* minDistIndex_dev;
	int* minDistValue_dev;
	cudaMalloc((void**)&d_dev, NUM_VERTICES * sizeof(int));
	cudaMalloc((void**)&visited_dev, NUM_VERTICES * sizeof(bool));
	cudaMalloc((void**)&minDistIndex_dev, sizeof(int));
	cudaMalloc((void**)&minDistValue_dev, sizeof(int));

	cudaMemcpy(d_dev, distancias, NUM_VERTICES * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(visited_dev, visitados, NUM_VERTICES * sizeof(bool), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Início da Contagem de Tempo
	cudaEventRecord(start);
	for (int count = 0; count < NUM_VERTICES - 1; count++) {
		int u = findNextVertex(d_dev, visited_dev, minDistIndex_dev, minDistValue_dev, NUM_VERTICES);
		if (u == -1) break;
		visitados[u] = true;

		struct No* v = grafo->cabeca[u];

		while (v != NULL) {
			if (!visitados[v->verticeDestino] &&
				distancias[u] + v->pesoAresta < distancias[v->verticeDestino]) {
				distancias[v->verticeDestino] = distancias[u] + v->pesoAresta;
			}
			v = v->proxNo;
		}

		cudaMemcpy(d_dev, distancias, NUM_VERTICES * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(visited_dev, visitados, NUM_VERTICES * sizeof(bool), cudaMemcpyHostToDevice);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Tempo de execucao da funcao dijkstra_CUDA: %.6f ms\n", milliseconds);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// Fim da contagem de tempo

	cudaFree(d_dev);
	cudaFree(visited_dev);
	cudaFree(minDistIndex_dev);
	cudaFree(minDistValue_dev);

	printf("\nDistancias minimas a partir do vertice %d:\n", inicio);
	for (int i = 0; i < NUM_VERTICES; i++) {
		printf("Vertice %d: %d\n", i, distancias[i]);
	}
}


//FUNÇÃO PRINCIPAL
int main(void) {

	struct Grafo* grafo = criarGrafo(NUM_VERTICES);
	int numArestas = 0;
	int vertice_de_entrada = 0;

	//const char* grafo16    = "D:\\Grafos\\grafo.txt";
	const char* grafo256 = "D:\\Grafos\\grafo256.txt";
	//const char* grafo512   = "D:\\Grafos\\grafo512.txt";
	//const char* grafo1024  = "D:\\Grafos\\grafo1024.txt";
	//const char* grafo2048  = "D:\\Grafos\\grafo2048.txt";
	//const char* grafo4096  = "D:\\Grafos\\grafo4096.txt";
	//const char* grafo8192  = "D:\\Grafos\\grafo8192.txt";
	//const char* grafo16384 = "D:\\Grafos\\grafo16384.txt";

	//Cálculo do Tamanho do Grafo
	for (int i = 0; i < NUM_VERTICES; i++) {
		for (int j = i + 1; j < NUM_VERTICES; j++) {
			numArestas++;
		}
	}

	printf("Numero de Vertices = %d\n", NUM_VERTICES);
	printf("Numero de Arestas = %d\n", numArestas);

	grafo = carregarGrafo(grafo256);

	//imprimirGrafo(grafo);

	//EXECUÇÃO DO ALGORITMO DE DIJKSTRA PARALELO
	for (int m = 0; m < 30; m++)
		dijkstra_CUDA(grafo, vertice_de_entrada++);

	free(grafo);

	return 0;
}