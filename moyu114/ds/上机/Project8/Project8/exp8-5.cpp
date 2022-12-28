#include"graph.h"
void Prim(MatGraph g, int v) {
	int lowcost[MAXV];
	int min;
	int closest[MAXV], i, j, k;
	for (i = 0;i < g.n;i++)//给lowcost[]和closest[]置初值
	{
		lowcost[i] = g.edges[v][i];
		closest[i] = v;
	}
	for (i = 1;i < g.n;i++) {//输出(n-1)条边
		min = INF;
		for (j = 0;j < g.n;j++) {//在(V-U)中找出离U最近的顶点k
			if (lowcost[j] != 0 && lowcost[j] < min) {
				min = lowcost[j];
				k = j;//k记录最近顶点编号
			}
		}
		printf(" 边(%d，%d)权为:%d\n", closest[k], k, min);
		lowcost[k] = 0;		//标记k已经加入U
		for (j = 0;j < g.n;j++)	//修改数组lowcost和closest
			if (lowcost[j] != 0 && g.edges[k][j] < lowcost[j])//寻找第j列cost最少的
			{
				lowcost[j] = g.edges[k][j];     //将其记录
				closest[j] = k;         //记录其顶点编号
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
	cout << "邻接矩阵为：" << endl;
	DispMat(g);
	cout << "Prim算法结果为：" << endl;
	Prim(g, 0);
}
