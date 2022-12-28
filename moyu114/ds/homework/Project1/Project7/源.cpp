#include<cstdio>
#include<iostream>
#include<vector>
using namespace std;
void Ctree() {
	hello();
}
void Ctree(BTNode* b, SqBTree a, int i)
{
	if (b != NULL)
	{
		a[i] = b->data;
		Ctree(b->lchild, a, 2 * i);
		Ctree(b->rchild, a, 2 * i + 1);
	}
	else a[i] = '#';
}

void FindMinNode(BTNode* b, char& min)
{
	if (b->data < min)
		min = b->data;
	FindMinNode(b->lchild, min); //��������������С���ֵ
	FindMinNode(b->rchild, min); //��������������С���ֵ
}
void MinNode(BTNode* b) //�����С���ֵ
{
	if (b != NULL)
	{
		char min = b->data;
		FindMinNode(b, min);
		cout<<"Min="<<min<<endl;
	}
}
void PrintNode(BTNode* p) //�����pΪ����������
{
	if (p != NULL)
	{
		cout<<p->data;
		PrintNode(p->lchild);
		PrintNode(p->rchild);
	}
}
void X_Child(BTNode* b, char x) //���x��������
{
	if (b != NULL)
	{
		if (b->data == x)
		{
			if (b->lchild != NULL)
				PrintNode(b->lchild);
			if (b->rchild != NULL)
				PrintNode(b->rchild);
			return;
		}
		X_Child(b->lchild, x);
		X_Child(b->rchild, x);
	}
}
bool CompBTree(BTNode* b)
{
	BTNode* Qu[MaxSize], * p; //����һ�����У����ڲ�α���
	int front = 0, rear = 0; //���ζ��еĶ�ͷ��βָ��
	bool cm = true; //cmΪ���ʾ������Ϊ��ȫ������
	bool bj = true; //bjΪ���ʾ��ĿǰΪֹ���н��������Һ���
	if (b == NULL) return true; //���������������ȫ������
	rear++;
	Qu[rear] = b; //��������
	while (front != rear) //���в���
	{
		front = (front + 1) % MaxSize;
		p = Qu[front]; //���ӽ��p
		if (p->lchild == NULL) //p���û������
		{
			bj = false; //���ֽ��pȱ���ӵ����
			if (p->rchild != NULL) //û�����ӵ����Һ���,Υ��(1),
				cm = false;
		}
		else //p���������
		{
			if (!bj) cm = false; //bjΪ�ٶ����p�������ӣ�Υ��(2)
			rear = (rear + 1) % MaxSize;
			Qu[rear] = p->lchild; //���ӽ���
			if (p->rchild == NULL)
				bj = false; //���ֽ��pȱ�Һ��ӵ����
			else //p�����Һ���,������ж�
			{
				rear = (rear + 1) % MaxSize;
				Qu[rear] = p->rchild; //��p�����Һ��ӽ���
			}
		}
	}
	return cm;
}
void InDs1(MatGraph g) //���ͼG ��ÿ����������
{
	int i, j, n;
	printf("���������:\n");
	for (j = 0;j < g.n;j++)//���Ϊ�����е�j��Ԫ��֮��
	{
		n = 0;
		for (i = 0;i < g.n;i++)
			if (g.edges[i][j] != 0)
				n++; //n�ۼ������
		printf(" ����%d:%d\n", j, n);
	}
}
void OutDs1(MatGraph g) //���ͼG��ÿ������ĳ���
{
	int i, j, n;
	printf("���������:\n");
	for (i = 0;i < g.n;i++)//����Ϊ�����е�i��Ԫ��֮��
	{
		n = 0;
		for (j = 0;j < g.n;j++)
			if (g.edges[i][j] != 0)
				n++; //n�ۼƳ�����
		printf(" ����%d:%d\n", i, n);
	}
}
void ZeroOutDs1(MatGraph g) //���ͼG�г���Ϊ0�Ķ������
{
	int i, j, n;
	printf("����Ϊ0�Ķ���:");
	for (i = 0;i < g.n;i++)
	{
		n = 0;
		for (j = 0;j < g.n;j++)
			if (g.edges[i][j] != 0) //����һ������
				n++;
		if (n == 0)
			printf("%2d\n", i);
	}
	printf("\n");
}



void InDs2(AdjGraph* G) //���ͼG��ÿ����������
{
	ArcNode* p;
	int A[MAXV], i; //A��Ÿ���������
	for (i = 0;i < G->n;i++) //A��Ԫ���ó�ֵ0
		A[i] = 0;
	for (i = 0;i < G->n;i++) //ɨ������ͷ���
	{
		p = G->adjlist[i].firstarc;
		while (p != NULL) //ɨ��߽��
		{
			A[p->adjvex]++; //��ʾi��p->adjvex������һ����
			p = p->nextarc;
		}
	}
	printf("���������:\n"); //�������������
	for (i = 0;i < G->n;i++)
		printf(" ����%d:%d\n", i, A[i]);
}
void OutDs2(AdjGraph* G) //���ͼG��ÿ������ĳ���
{
	int i, n;
	ArcNode* p;
	printf("���������:\n");
	for (i = 0;i < G->n;i++) //ɨ������ͷ���
	{
		n = 0;
		p = G->adjlist[i].firstarc;
		while (p != NULL) //ɨ��߽��
		{
			n++; //�ۼƳ��ߵ���
			p = p->nextarc;
		}
		printf(" ����%d:%d\n", i, n);
	}
}
void ZeroOutDs2(AdjGraph* G) //���ͼG�г���Ϊ0�Ķ�����
{
	int i, n;
	ArcNode* p;
	printf("����Ϊ0�Ķ���:");
	for (i = 0;i < G->n;i++) //ɨ������ͷ���
	{
		p = G->adjlist[i].firstarc;
		n = 0;
		while (p != NULL) //ɨ��߽��
		{
			n++; //�ۼƳ��ߵ���
			p = p->nextarc;
		}
		if (n == 0) //���������Ϊ0�Ķ�����
			printf("%2d", i);
	}
	printf("\n");
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
	visited[u] = 0; //�ָ�����,ʹ�ö��������ʹ��
}
int main() {
	AdjGraph* G;
	int A[MAXV][MAXV] = {
	{0,4,6,6,INF,INF,INF}, 
	{INF,0,1,INF,7,INF,INF},
	{INF,INF,0,INF,6,4,INF},
	{INF,INF,2,0,INF,5,INF},
	{INF,INF,INF,INF,0,INF,6},
	{INF,INF,INF,INF,1,0,8},
	{INF,INF,INF,INF,INF,INF,0} };
		
	int n = 7, e = 12;
	CreateAdj(G, A, n, e); //�������̡̳���ͼ8.35���ڽӱ�
	printf("ͼG���ڽӱ�:\n");
	DispAdj(G); //����ڽӱ�
	int u = 0, v = 5;
	int path[MAXV];
	printf("��%d->%d������·��:\n", u, v);
	findpath(G, u, v, path, 0, 0);
	DestroyAdj(G);
	return 1;
}