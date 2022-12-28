#include<iostream>
using namespace std;
#define M 4
#define N 4
#define MAXSIZE 100
int maze[M + 2][N + 2]={
	{1,1,1,1,1,1},{1,0,0,0,1,1},{1,0,1,0,0,1},
	{1,0,0,0,1,1},{1,1,0,0,0,1},{1,1,1,1,1,1}};
struct {
	int i, j;
	int di;
}st[MAXSIZE], path[MAXSIZE];       //st���洢��ǰ·����path�洢���·��
int top = -1;                      //ջ��ָ��
int cnt = 1;                     //��¼·������
int minlen = MAXSIZE;              //��¼��С·������
void dispapath(){
	int k;
	printf("%5d:", cnt++);
	for (k = 0;k <= top;k++) {         //���·��
		printf("(%d,%d) ", st[k].i, st[k].j);
	}
	cout << endl;
	if (top + 1 < minlen) {            //Ѱ����С·��
		for (k = 0;k <= top;k++) {
			path[k] = st[k];
		}
		minlen = top + 1;
	}

}
void dispminpath() {
	cout << "��С·����" << endl;
	cout << "���ȣ�" <<minlen<< endl;
	cout << "·����";
	for (int k = 0;k < minlen;k++) {
		printf("(%d,%d)", path[k].i, path[k].j);
	}
	cout << endl;
}
void pathfind(int xi,int yi,int xe,int ye) {
	int i, j, il=1, jl=1, di;
	bool find;
	top++;                   //��ջ
	st[top].i = xi;
	st[top]. j = yi;
	st[top].di = -1;maze[xi][yi] = -1; //��ʼ����ֵ
	while (top > -1) {
		i = st[top].i;
		j = st[top].j;
		di = st[top].di;
		if (i == xe && j == ye) {    //�ҵ�����
			dispapath();
			maze[i][j] = 0;         //��Ϊ��Ѱը·����ʱ�������㣬������Ҫ�����ڱ�Ϊ����
			top--;                  //��ջ
			i = st[top].i;
			j = st[top].j;
			di = st[top].di;        //ջ����Ϊ��ǰ����
		}
		find = false;
		while (di < 4 && !find) {    //������Ѱ�ҿ��߷���
			di++;
			switch (di) {
			case 0:il = i - 1; jl = j; break;
			case 1:il = i; jl = j + 1; break;
			case 2:il = i + 1; jl = j; break;
			case 3:il = i; jl = j - 1; break;
			}
			if (maze[il][jl] == 0)find = true;
		}
		if (find) {                //�ҵ��˿��߷���
			st[top].di = di;       //�޸�ԭջ��Ԫ�ص�diֵ
			top++;
			st[top].i = il;
			st[top].j = jl;
			st[top].di = -1;      //��һ���߷�����ջ
			maze[il][jl] = -1;    //���߹���·��Ϊ������
		}
		else {
			maze[i][j] = 0;     //û·���ߣ���ջ��������λ����������Ϊ����
			top--;
		}
	}
	dispminpath();             //������·��
}
int main() {
	cout << "�Թ�����·����" << endl;
	pathfind(1, 1, M, N);
	return 0;
}