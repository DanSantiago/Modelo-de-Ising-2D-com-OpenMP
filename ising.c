#include "funcoes.h"

int main(){

	FILE *arq;
	int i, j, pos[2], dE=0;
	int Rede[L][L];												//rede quadrada
	int J = 1;                       							//energia de interação
	int k = 1; 													//constante de boltzman
	int B = 0;													//campo externo
	double T = 5;												//temperatura final
	double Tmin = 0.5;											//temperatura inicial
	double dT = 0.1;											//incremento na temperatura por iteração
	unsigned int Msteps = 100000000;									//número de iterações do método de Monte Carlo	
	double E=0, E2_media=0, E_media=0, E_total=0, E2_total=0;
	double M=0, M2_media=0, M_media=0, M_total=0, M2_total=0;
	double Mabs_media=0, Mabs_total=0;
	double calor_esp = 0, susc_mag = 0;
	
	unsigned int seed = 158235;      					//gera a semente para gerar números aleatórios
	
	arq = fopen("dados.dat", "w");						//arquivo para armazenagem de dados

	//inicia a rede aleatóriamente
	inicia_malha(&seed, Rede);
	
	//loop da temperatura
	for(;T>=Tmin;T-=dT)
	{
		//termalização
		equilibra(&seed, J, B, k, T, Rede);
		
		//observáveis com valores no equilíbrio 
		M = magnetizacao_total(Rede);
		E = energia_total(J, B, Rede);
		
		E_total=0;
		E2_total=0;
		M_total=0;
		M2_total=0;
		Mabs_total=0;

		#pragma omp parallel for schedule(dynamic, 12500000) private(dE, pos) \
		shared(J, B, k, T) firstprivate(E, M, Rede) reduction(+: E_total, E2_total, M_total, M2_total, Mabs_total)
			//loop do Monte Carlo
			for(i=1;i<=Msteps;i++){
				//loop de Metropolis
				for(j=1;j<=N;j++){
					escolhe_pos(pos, &seed);
						
					if(testa_flip(pos, &dE, &seed, J, B, k, T, Rede)){
						//ajusta os observáveis
						E+=2*dE;
						M+=2*Rede[pos[0]][pos[1]];
					}
				}
					
				//soma dos observavéis
				E_total+=E;
				E2_total+= E*E;
				M_total+=M;
				M2_total+= M*M;
				Mabs_total+=abs(M);
			}			
		
		//média dos observáveis
		E_media=(E_total/(Msteps*N))*0.5;         		  			//<E> - fator 1/2 pela contagem dupla dos pares
		E2_media=(E2_total/(Msteps*N))*0.25;	 		  			//<E²>	- fator 1/4 idem (1/2*1/2)
		M_media=M_total/(Msteps*N);	 		  						//<M>
		M2_media=M2_total/(Msteps*N);	 		  					//<M²>
		Mabs_media=Mabs_total/(Msteps*N);							//<|M|>
		calor_esp = (E2_media-(E_media*E_media*N))/(k*T*T);     	//C_v = (<E²> - <E>²*N)/(k*T²) | <E>² multiplicado N porque <E>² vai ter um N² no denominador
		susc_mag = (M2_media-(M_media*M_media*N))/(k*T); 			//X = (<M²> - <M>²*N)/(k*T) | <M>² multiplicado N porque <M>² vai ter um N² no denominador

		fprintf(arq, "%.2lf %lf %lf %lf %lf\n", T, Mabs_media, E_media, calor_esp, susc_mag);
	}
	
	fclose(arq);
	printf("Fim do programa.\nObserváveis salvos no arquivo 'dados.dat'\nTabelas na ordem: Temperatura | <Magnetização Absoluta> p/ spin | <Energia> p/ spin | Calor Específico p/ spin | Susceptibilidade Magnética p/ spin\n");
	
	return 0;
}

