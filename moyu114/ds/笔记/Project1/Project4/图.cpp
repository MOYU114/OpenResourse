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
//�����������
typedef struct 
{
	int no;
	InfoType info;
}VertexType;
//�ڽӾ���
typedef struct 
{
	int edges[MAXV][MAXV];
	int n, e;
	VertexType vexs[MAXV];
}MatGraph;
//�ڽӱ�
typedef struct ANode
{
	int adjvex;			//�ñߵ��յ���
	struct ANode* nextarc;	//ָ����һ���ߵ�ָ��
	InfoType weight;		//�ñߵ�Ȩֵ����Ϣ
}  ArcNode;

typedef struct Vnode
{
	VertexType data;			//������Ϣ
	ArcNode* firstarc;		//ָ���һ����
}  VNode;

typedef struct
{
	VNode adjlist[MAXV];	//�ڽӱ�
	int n,e;			//ͼ�ж�����n�ͱ���e
} AdjGraph;

void CreateAdj(AdjGraph*& G, int A[MAXV][MAXV], int n, int e) {
	int i, j=0;
	ArcNode* p;
	G = (AdjGraph*)malloc(sizeof AdjGraph);
	for (i = 0;i < n;i++) {
		G->adjlist[i].firstarc = NULL;//���ڽӱ�������ͷ����ָ�����ó�ֵ
	}
	for (i = 0;i < n;i++) {
		for (j = n - 1;j >= 0;j--) {
			if (A[i][j] != 0 && A[i][j] != INF) { //����һ����
				p = (ArcNode*)malloc(sizeof ArcNode);//����һ�����p
				p->adjvex = j;                     //����ڽӵ�
				p->weight = A[i][j];              //���Ȩ
				p->nextarc = G->adjlist[i].firstarc;//����ͷ�巨������p
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
		printf("��\n");
	}
}
void DestroyAdj(AdjGraph*& G) {
	int i;
	ArcNode* pre, * p;

	for (i = 0;i < G->n;i++) {//ɨ�����еĵ�����
		pre = G->adjlist[i].firstarc;//pָ���i����������׽��
		if (pre != NULL) {
			p = pre->nextarc;
			while (p != NULL)	//�ͷŵ�i������������б߽��
			{
				free(pre);
				pre = p; p = p->nextarc;
			}
			free(pre);
		}
	}
	free(G);//�ͷ�ͷ�������
}
void MatToList(MatGraph g, AdjGraph*& G) {
	//���ڽӾ���gת�����ڽӱ�G
	int i, j;
	ArcNode* p;
	G = (AdjGraph*)malloc(sizeof(AdjGraph));
	for (i = 0;i < G->n;i++) {//���ڽӱ�������ͷ����ָ�����ó�ֵ
		G->adjlist[i].firstarc = NULL;
	}
	for (i = 0;i < g.n;i++) {//����ڽӾ�����ÿ��Ԫ��
		for (j = g.n - 1;j >= 0;j--) {
			if (g.edges[i][j] != 0 && g.edges[i][j] != INF) {//����һ����
				p = (ArcNode*)malloc(sizeof(ArcNode));//��һ���߽��p
				p->adjvex = j;p->weight = g.edges[i][j];
				p->nextarc = G->adjlist[i].firstarc;;     //����ͷ�巨������p
				G->adjlist[i].firstarc = p;
			}
		}
	}
	G->n = g.n;G->e = g.e;
}
void ListToMat(AdjGraph* G, MatGraph& g)
//���ڽӱ�Gת�����ڽӾ���g
{
	int i;
	ArcNode* p;
	for (i = 0;i < G->n;i++)			//ɨ�����еĵ�����
	{
		p = G->adjlist[i].firstarc;		//pָ���i����������׽��
		while (p != NULL)		//ɨ���i��������
		{
			g.edges[i][p->adjvex] = 1;//��vi���ܹ������·����һ
			p = p->nextarc;
		}
	}
	g.n = G->n; g.e = G->e;
}
int visited[MAXV];
void findpath(AdjGraph* G, int u, int v, int path[], int d, int length)
{ //d��ʾpath�ж����������ʼΪ0��length��ʾ·�����ȣ���ʼΪ0
	int w, i;
	ArcNode* p;
	path[d] = u; d++; //����u���뵽·���У�d��1
	visited[u] = 1; //���ѷ��ʱ��
	if (u == v && d > 0) //�ҵ�һ��·�������
	{
		printf(" ·������:%d, ·��:", length);
		for (i = 0;i < d;i++)
			printf("%2d", path[i]);
		printf("\n");
	}
	p = G->adjlist[u].firstarc; //pָ�򶥵�u�ĵ�һ���ڽӵ�
	while (p != NULL)
	{
		w = p->adjvex; //wΪ����u���ڽӵ�
		if (visited[w] == 0) //��w����δ����,�ݹ������
			findpath(G, w, v, path, d, p->weight + length);
		p = p->nextarc; //pָ�򶥵�u����һ���ڽӵ�
	}
	visited[u] = 0;
}
int visit[100];
void DFS(AdjGraph* G, int v) {
	ArcNode* p;int w;
	visit[v] = 1;//���ѷ��ʱ��
	cout << v;//��������ʶ���ı��
	p = G->adjlist[v].firstarc;//pָ�򶥵�v�ĵ�һ���ߵı�ͷ���
	while (p != NULL) {
		w = p->adjvex;
		if (visit[w] == 0)//��w����δ���ʣ��ݹ������
			DFS(G, w);
		p = p->nextarc;//pָ�򶥵�v����һ���ߵı�ͷ���
	}

}
void BFS(AdjGraph* G, int v) {
	int w, i;
	ArcNode* p;
	SqQueue* qu;//���廷�ζ���ָ��
	InitQueue(qu);//��ʼ������
	int Visit[MAXV];//���嶥����ʱ������
	for (i = 0;i < G->n;i++)
		Visit[i] = 0;//���ʱ�������ʼ��
	cout << v;//��������ʶ���ı��
	Visit[v] = 1;//���ѷ��ʱ��
	enQueue(qu, v);
	while (!QueueEmpty(qu)) {//�Ӳ���ѭ��
		deQueue(qu, w);//����һ������w
		p = G->adjlist->firstarc;//ָ��w�ĵ�һ���ڽӵ�
		while (p != NULL) {//����w�������ڽӵ�
			if (Visit[p->adjvex] == 0) {//����ǰ�ڽӵ�δ������
				cout << p->adjvex;//���ʸ��ڽӵ�
				Visit[p->adjvex] = 1;//���ѷ��ʱ��
				enQueue(qu, p->adjvex);//�ö������
			}
			p = p->nextarc;//����һ���ڽӵ�
		}
	}
	cout << endl;
}

void Prim(MatGraph g,int v) {
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
			if (lowcost[j] != 0 &&lowcost[j] < min) {
				min = lowcost[j];
				k = j;//k��¼���������
			}
		}
		printf(" ��(%d��%d)ȨΪ:%d\n",closest[k],k,min);
		lowcost[k] = 0;		//���k�Ѿ�����U
		for (j = 0;j < g.n;j++)	//�޸�����lowcost��closest
			if (lowcost[j] != 0 && g.edges[k][j] < lowcost[j])//Ѱ�ҵ�j��cost���ٵ�
			{
				lowcost[j] = g.edges[k][j];     //�����¼
				closest[j] = k;         //��¼�䶥����
			}
	}
}

typedef struct
{
	int u;     //�ߵ���ʼ����
	int v;      //�ߵ���ֹ����
	int w;     //�ߵ�Ȩֵ
} Edge;

void Kruskal(MatGraph g) {
	int i, j, u1, v1, sn1, sn2, k;
	int vset[MAXV];
	Edge E[MAXSIZE];//������б�
	k = 0;//E������±��0��ʼ��
	for (i = 0;i < g.n;i++) {//��g�����ı߼�E
		for (j = 0;j < g.n;j++) {
				if (g.edges[i][j] != 0 && g.edges[i][j] != INF) {
					E[k].u = i;E[k].v = j;E[k].w = g.edges[i][j];
					k++;
				}
			}
	}
	
	Insertsort(E, g.e);//��ֱ�Ӳ��������E���鰴Ȩֵ��������
	for (i - 0;i < g.n;i++)//��ʼ����������
		vset[i] = i;
	k = 1;		//k��ʾ��ǰ�����������ĵڼ����ߣ���ֵΪ1
	j = 0;		//E�бߵ��±꣬��ֵΪ0
	while (k < g.n)	//���ɵı���С��nʱѭ��
	{
		u1 = E[j].u;v1 = E[j].v;	//ȡһ���ߵ�ͷβ����
		sn1 = vset[u1];
		sn2 = vset[v1];		//�ֱ�õ��������������ļ��ϱ��
		if (sn1 != sn2)  	//���������ڲ�ͬ�ļ���
		{
			printf("  (%d��%d):%d\n",u1,v1,E[j].w);
			k++;		   	//���ɱ�����1
			for (i = 0;i < g.n;i++)  	//��������ͳһ���
				if (vset[i] == sn2) 	//���ϱ��Ϊsn2�ĸ�Ϊsn1
					vset[i] = sn1;
		}
		j++;			   //ɨ����һ����
	}
}

void Dijkstra(MatGraph g��int v)
{
	int dist[MAXV]��path[MAXV];
	int s[MAXV];
	int mindis, i, j, u;
	for (i = 0;i < g.n;i++)
	{
		dist[i] = g.edges[v][i];	//�����ʼ��
		s[i] = 0;			//s[]�ÿ�
		if (g.edges[v][i] < INF)	//·����ʼ��
			path[i] = v;		//����v��i�б�ʱ
		else
			path[i] = -1;		//����v��iû��ʱ
	}
	s[v] = 1;	 		//Դ��v����S��
	for (i = 0;i < g.n;i++)	 	//ѭ��n-1��
	{
		mindis = INF;
		for (j = 0;j < g.n;j++)
			if (s[j] == 0 && dist[j] < mindis)
			{
				u = j;
				mindis = dist[j];
			}
		s[u] = 1;			//����u����S��
		for (j = 0;j < g.n;j++)	//�޸Ĳ���s�еĶ���ľ���
			if (s[j] == 0)
				if (g.edges[u][j] < INF && dist[u] + g.edges[u][j] < dist[j])
				{
					dist[j] = dist[u] + g.edges[u][j];
					path[j] = u;
				}
	}
	Dispath(dist, path, s, g.n, v);	//������·��
}

void Floyd(MatGraph g)		//��ÿ�Զ���֮������·��
{
	int A[MAXVEX][MAXVEX];	//����A����
	int path[MAXVEX][MAXVEX];	//����path����
	int i, j, k;
	for (i = 0;i < g.n;i++)
		for (j = 0;j < g.n;j++)
		{
			A[i][j] = g.edges[i][j];
			if (i != j && g.edges[i][j] < INF)
				path[i][j] = i; 	//i��j����֮����һ����ʱ
			else			 //i��j����֮��û��һ����ʱ
				path[i][j] = -1;
		}
	for (k = 0;k < g.n;k++)		//��Ak[i][j]
	{
		for (i = 0;i < g.n;i++)
			for (j = 0;j < g.n;j++)
				if (A[i][j] > A[i][k] + A[k][j])	//�ҵ�����·��
				{
					A[i][j] = A[i][k] + A[k][j];	//�޸�·������
					path[i][j] = path[k][j]; 	//�޸����·��Ϊ��������k
				}
	}
}
