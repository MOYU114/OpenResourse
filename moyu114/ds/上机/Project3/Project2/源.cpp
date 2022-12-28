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
}st[MAXSIZE], path[MAXSIZE];       //st来存储当前路径，path存储最短路径
int top = -1;                      //栈顶指针
int cnt = 1;                     //记录路径长度
int minlen = MAXSIZE;              //记录最小路径长度
void dispapath(){
	int k;
	printf("%5d:", cnt++);
	for (k = 0;k <= top;k++) {         //输出路径
		printf("(%d,%d) ", st[k].i, st[k].j);
	}
	cout << endl;
	if (top + 1 < minlen) {            //寻找最小路径
		for (k = 0;k <= top;k++) {
			path[k] = st[k];
		}
		minlen = top + 1;
	}

}
void dispminpath() {
	cout << "最小路径：" << endl;
	cout << "长度：" <<minlen<< endl;
	cout << "路径：";
	for (int k = 0;k < minlen;k++) {
		printf("(%d,%d)", path[k].i, path[k].j);
	}
	cout << endl;
}
void pathfind(int xi,int yi,int xe,int ye) {
	int i, j, il=1, jl=1, di;
	bool find;
	top++;                   //入栈
	st[top].i = xi;
	st[top]. j = yi;
	st[top].di = -1;maze[xi][yi] = -1; //初始化数值
	while (top > -1) {
		i = st[top].i;
		j = st[top].j;
		di = st[top].di;
		if (i == xe && j == ye) {    //找到出口
			dispapath();
			maze[i][j] = 0;         //因为在寻炸路径的时候将其置零，所以需要将出口变为可走
			top--;                  //出栈
			i = st[top].i;
			j = st[top].j;
			di = st[top].di;        //栈顶变为当前方块
		}
		find = false;
		while (di < 4 && !find) {    //操作，寻找可走方块
			di++;
			switch (di) {
			case 0:il = i - 1; jl = j; break;
			case 1:il = i; jl = j + 1; break;
			case 2:il = i + 1; jl = j; break;
			case 3:il = i; jl = j - 1; break;
			}
			if (maze[il][jl] == 0)find = true;
		}
		if (find) {                //找到了可走方块
			st[top].di = di;       //修改原栈顶元素的di值
			top++;
			st[top].i = il;
			st[top].j = jl;
			st[top].di = -1;      //下一可走方块入栈
			maze[il][jl] = -1;    //将走过的路变为不可走
		}
		else {
			maze[i][j] = 0;     //没路可走，退栈，并将该位置重新设置为可走
			top--;
		}
	}
	dispminpath();             //输出最短路径
}
int main() {
	cout << "迷宫所有路径：" << endl;
	pathfind(1, 1, M, N);
	return 0;
}