#include"graph.h"
void Prim(MatGraph g, int v) {
	int lowcost[MAXV];
	int min;
	int closest[MAXV], i, j, k;
	for (i = 0;i < g.n;i++)//��lowcost[]��closest[]�ó�ֵ
	{
		lowcost[i] = g.edges[v][i];
		closest[i] = v;
	}
	for (i = 1;i < g.n;i++) {//���(n-1)����
		min = INF;
		for (j = 0;j < g.n;j++) {//��(V-U)���ҳ���U����Ķ���k
			if (lowcost[j] != 0 && lowcost[j] < min) {
				min = lowcost[j];
				k = j;//k��¼���������
			}
		}
		printf(" ��(%d��%d)ȨΪ:%d\n", closest[k], k, min);
		lowcost[k] = 0;		//���k�Ѿ�����U
		for (j = 0;j < g.n;j++)	//�޸�����lowcost��closest
			if (lowcost[j] != 0 && g.edges[k][j] < lowcost[j])//Ѱ�ҵ�j��cost���ٵ�
			{
				lowcost[j] = g.edges[k][j];     //�����¼
				closest[j] = k;         //��¼�䶥����
			}
	}
}
int main() {
	int v = 3;
	MatGraph g;
	int A[MAXV][MAXV] = {
		{0,5,INF,7,INF,3},{5,0,4,INF,INF,INF},
		{8,4,0,5,INF,9},{7,INF,5,0,5,6},
		{INF,INF,INF,5,0,1},{3,INF,9,6,1,0} };
	int n = 6, e = 10;
	CreateMat(g, A, n, e);
	cout << "�ڽӾ���Ϊ��" << endl;
	DispMat(g);
	cout << "Prim�㷨���Ϊ��" << endl;
	Prim(g, 0);
}
