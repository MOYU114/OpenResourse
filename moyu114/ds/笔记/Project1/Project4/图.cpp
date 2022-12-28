#include<iostream>
#include<algorithm>
using namespace std;
#define MAXV 50
#define INF 32767
typedef char ElemType;
#define MAXSIZE 50
typedef struct {
	ElemType data[MAXSIZE];
	int front, rear;
}SqQueue;

void InitQueue(SqQueue*& q) {
	q = (SqQueue*)malloc(sizeof SqQueue);
	q->front = q->rear = -1;
}
void DestoryQueue(SqQueue*& q) {
	free(q);
}
bool QueueEmpty(SqQueue* q) {
	return(q->front == q->rear);
}
bool enQueue(SqQueue*& q, ElemType e) {
	if (q->rear == MAXSIZE - 1)return false;
	q->rear++;
	q->data[q->rear] = e;
	return true;
}
bool deQueue(SqQueue*& q, ElemType e) {
	if (q->front == q->rear)return false;
	q->front++;
	e = q->data[q->front];
	return true;
}












typedef char InfoType;
//顶点基本类型
typedef struct 
{
	int no;
	InfoType info;
}VertexType;
//邻接矩阵
typedef struct 
{
	int edges[MAXV][MAXV];
	int n, e;
	VertexType vexs[MAXV];
}MatGraph;
//邻接表
typedef struct ANode
{
	int adjvex;			//该边的终点编号
	struct ANode* nextarc;	//指向下一条边的指针
	InfoType weight;		//该边的权值等信息
}  ArcNode;

typedef struct Vnode
{
	VertexType data;			//顶点信息
	ArcNode* firstarc;		//指向第一条边
}  VNode;

typedef struct
{
	VNode adjlist[MAXV];	//邻接表
	int n,e;			//图中顶点数n和边数e
} AdjGraph;

void CreateAdj(AdjGraph*& G, int A[MAXV][MAXV], int n, int e) {
	int i, j=0;
	ArcNode* p;
	G = (AdjGraph*)malloc(sizeof AdjGraph);
	for (i = 0;i < n;i++) {
		G->adjlist[i].firstarc = NULL;//给邻接表中所有头结点的指针域置初值
	}
	for (i = 0;i < n;i++) {
		for (j = n - 1;j >= 0;j--) {
			if (A[i][j] != 0 && A[i][j] != INF) { //存在一条边
				p = (ArcNode*)malloc(sizeof ArcNode);//创建一个结点p
				p->adjvex = j;                     //存放邻接点
				p->weight = A[i][j];              //存放权
				p->nextarc = G->adjlist[i].firstarc;//采用头插法插入结点p
				G->adjlist[i].firstarc = p;
			}
			
		}
		
	}
	G->n = n;G->e = e;
}
void DispAdj(AdjGraph* G) {
	int i;
	ArcNode* p;
	for (i = 0;i < G->n;i++) {
		p = G->adjlist[i].firstarc;
		printf("%3d: ", i);
		while (p != NULL) {
			printf("%3d[%d]-> ", p->adjvex, p->weight);
			p = p->nextarc;
		}
		printf("∧\n");
	}
}
void DestroyAdj(AdjGraph*& G) {
	int i;
	ArcNode* pre, * p;

	for (i = 0;i < G->n;i++) {//扫描所有的单链表
		pre = G->adjlist[i].firstarc;//p指向第i个单链表的首结点
		if (pre != NULL) {
			p = pre->nextarc;
			while (p != NULL)	//释放第i个单链表的所有边结点
			{
				free(pre);
				pre = p; p = p->nextarc;
			}
			free(pre);
		}
	}
	free(G);//释放头结点数组
}
void MatToList(MatGraph g, AdjGraph*& G) {
	//将邻接矩阵g转换成邻接表G
	int i, j;
	ArcNode* p;
	G = (AdjGraph*)malloc(sizeof(AdjGraph));
	for (i = 0;i < G->n;i++) {//将邻接表中所有头结点的指针域置初值
		G->adjlist[i].firstarc = NULL;
	}
	for (i = 0;i < g.n;i++) {//检查邻接矩阵中每个元素
		for (j = g.n - 1;j >= 0;j--) {
			if (g.edges[i][j] != 0 && g.edges[i][j] != INF) {//存在一条边
				p = (ArcNode*)malloc(sizeof(ArcNode));//建一个边结点p
				p->adjvex = j;p->weight = g.edges[i][j];
				p->nextarc = G->adjlist[i].firstarc;;     //采用头插法插入结点p
				G->adjlist[i].firstarc = p;
			}
		}
	}
	G->n = g.n;G->e = g.e;
}
void ListToMat(AdjGraph* G, MatGraph& g)
//将邻接表G转换成邻接矩阵g
{
	int i;
	ArcNode* p;
	for (i = 0;i < G->n;i++)			//扫描所有的单链表
	{
		p = G->adjlist[i].firstarc;		//p指向第i个单链表的首结点
		while (p != NULL)		//扫描第i个单链表
		{
			g.edges[i][p->adjvex] = 1;//将vi中能够到达的路径置一
			p = p->nextarc;
		}
	}
	g.n = G->n; g.e = G->e;
}
int visited[MAXV];
void findpath(AdjGraph* G, int u, int v, int path[], int d, int length)
{ //d表示path中顶点个数，初始为0；length表示路径长度，初始为0
	int w, i;
	ArcNode* p;
	path[d] = u; d++; //顶点u加入到路径中，d增1
	visited[u] = 1; //置已访问标记
	if (u == v && d > 0) //找到一条路径则输出
	{
		printf(" 路径长度:%d, 路径:", length);
		for (i = 0;i < d;i++)
			printf("%2d", path[i]);
		printf("\n");
	}
	p = G->adjlist[u].firstarc; //p指向顶点u的第一个邻接点
	while (p != NULL)
	{
		w = p->adjvex; //w为顶点u的邻接点
		if (visited[w] == 0) //若w顶点未访问,递归访问它
			findpath(G, w, v, path, d, p->weight + length);
		p = p->nextarc; //p指向顶点u的下一个邻接点
	}
	visited[u] = 0;
}
int visit[100];
void DFS(AdjGraph* G, int v) {
	ArcNode* p;int w;
	visit[v] = 1;//置已访问标记
	cout << v;//输出被访问顶点的编号
	p = G->adjlist[v].firstarc;//p指向顶点v的第一条边的边头结点
	while (p != NULL) {
		w = p->adjvex;
		if (visit[w] == 0)//若w顶点未访问，递归访问它
			DFS(G, w);
		p = p->nextarc;//p指向顶点v的下一条边的边头结点
	}

}
void BFS(AdjGraph* G, int v) {
	int w, i;
	ArcNode* p;
	SqQueue* qu;//定义环形队列指针
	InitQueue(qu);//初始化队列
	int Visit[MAXV];//定义顶点访问标记数组
	for (i = 0;i < G->n;i++)
		Visit[i] = 0;//访问标记数组初始化
	cout << v;//输出被访问顶点的编号
	Visit[v] = 1;//置已访问标记
	enQueue(qu, v);
	while (!QueueEmpty(qu)) {//队不空循环
		deQueue(qu, w);//出队一个顶点w
		p = G->adjlist->firstarc;//指向w的第一个邻接点
		while (p != NULL) {//查找w的所有邻接点
			if (Visit[p->adjvex] == 0) {//若当前邻接点未被访问
				cout << p->adjvex;//访问该邻接点
				Visit[p->adjvex] = 1;//置已访问标记
				enQueue(qu, p->adjvex);//该顶点进队
			}
			p = p->nextarc;//找下一个邻接点
		}
	}
	cout << endl;
}

void Prim(MatGraph g,int v) {
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
			if (lowcost[j] != 0 &&lowcost[j] < min) {
				min = lowcost[j];
				k = j;//k记录最近顶点编号
			}
		}
		printf(" 边(%d，%d)权为:%d\n",closest[k],k,min);
		lowcost[k] = 0;		//标记k已经加入U
		for (j = 0;j < g.n;j++)	//修改数组lowcost和closest
			if (lowcost[j] != 0 && g.edges[k][j] < lowcost[j])//寻找第j列cost最少的
			{
				lowcost[j] = g.edges[k][j];     //将其记录
				closest[j] = k;         //记录其顶点编号
			}
	}
}

typedef struct
{
	int u;     //边的起始顶点
	int v;      //边的终止顶点
	int w;     //边的权值
} Edge;

void Kruskal(MatGraph g) {
	int i, j, u1, v1, sn1, sn2, k;
	int vset[MAXV];
	Edge E[MAXSIZE];//存放所有边
	k = 0;//E数组的下标从0开始计
	for (i = 0;i < g.n;i++) {//由g产生的边集E
		for (j = 0;j < g.n;j++) {
				if (g.edges[i][j] != 0 && g.edges[i][j] != INF) {
					E[k].u = i;E[k].v = j;E[k].w = g.edges[i][j];
					k++;
				}
			}
	}
	
	Insertsort(E, g.e);//用直接插入排序对E数组按权值递增排序
	for (i - 0;i < g.n;i++)//初始化辅助数组
		vset[i] = i;
	k = 1;		//k表示当前构造生成树的第几条边，初值为1
	j = 0;		//E中边的下标，初值为0
	while (k < g.n)	//生成的边数小于n时循环
	{
		u1 = E[j].u;v1 = E[j].v;	//取一条边的头尾顶点
		sn1 = vset[u1];
		sn2 = vset[v1];		//分别得到两个顶点所属的集合编号
		if (sn1 != sn2)  	//两顶点属于不同的集合
		{
			printf("  (%d，%d):%d\n",u1,v1,E[j].w);
			k++;		   	//生成边数增1
			for (i = 0;i < g.n;i++)  	//两个集合统一编号
				if (vset[i] == sn2) 	//集合编号为sn2的改为sn1
					vset[i] = sn1;
		}
		j++;			   //扫描下一条边
	}
}

void Dijkstra(MatGraph g，int v)
{
	int dist[MAXV]，path[MAXV];
	int s[MAXV];
	int mindis, i, j, u;
	for (i = 0;i < g.n;i++)
	{
		dist[i] = g.edges[v][i];	//距离初始化
		s[i] = 0;			//s[]置空
		if (g.edges[v][i] < INF)	//路径初始化
			path[i] = v;		//顶点v到i有边时
		else
			path[i] = -1;		//顶点v到i没边时
	}
	s[v] = 1;	 		//源点v放入S中
	for (i = 0;i < g.n;i++)	 	//循环n-1次
	{
		mindis = INF;
		for (j = 0;j < g.n;j++)
			if (s[j] == 0 && dist[j] < mindis)
			{
				u = j;
				mindis = dist[j];
			}
		s[u] = 1;			//顶点u加入S中
		for (j = 0;j < g.n;j++)	//修改不在s中的顶点的距离
			if (s[j] == 0)
				if (g.edges[u][j] < INF && dist[u] + g.edges[u][j] < dist[j])
				{
					dist[j] = dist[u] + g.edges[u][j];
					path[j] = u;
				}
	}
	Dispath(dist, path, s, g.n, v);	//输出最短路径
}

void Floyd(MatGraph g)		//求每对顶点之间的最短路径
{
	int A[MAXVEX][MAXVEX];	//建立A数组
	int path[MAXVEX][MAXVEX];	//建立path数组
	int i, j, k;
	for (i = 0;i < g.n;i++)
		for (j = 0;j < g.n;j++)
		{
			A[i][j] = g.edges[i][j];
			if (i != j && g.edges[i][j] < INF)
				path[i][j] = i; 	//i和j顶点之间有一条边时
			else			 //i和j顶点之间没有一条边时
				path[i][j] = -1;
		}
	for (k = 0;k < g.n;k++)		//求Ak[i][j]
	{
		for (i = 0;i < g.n;i++)
			for (j = 0;j < g.n;j++)
				if (A[i][j] > A[i][k] + A[k][j])	//找到更短路径
				{
					A[i][j] = A[i][k] + A[k][j];	//修改路径长度
					path[i][j] = path[k][j]; 	//修改最短路径为经过顶点k
				}
	}
}
