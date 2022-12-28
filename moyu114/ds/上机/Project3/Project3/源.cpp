#include<iostream>
#include<stdlib.h>
#define MAXSIZE 100
using namespace std;
typedef struct {
	int col[MAXSIZE];         //col[i]为皇后位置标识
	int top;                  //栈顶指针
}StackType;
void dispasolution(StackType st) {    //输出一个解
	static int cnt = 0;
	printf("第%d个解：",++cnt);
	for (int i = 1;i <= st.top;i++)
		printf("(%d,%d)", i, st.col[i]);
	printf("\n");
}
bool place(StackType st, int k, int j) { //遍历整个数组，看是否有冲突
	int i = 1;
	if (k == 1)return true;             //放一个皇后
	while (i <= k - 1) {                //看是否与前面的皇后有冲突
		if ((st.col[i] == j) || (abs(j - st.col[i]) == abs(i - k)))
			return false;
		i++;
	}
	return true;
}
void queen(int n) {
	int k;
	bool find;
	StackType st;
	st.top = 0;                     //从第一行开始，初始标号为0
	st.top++;
	st.col[st.top] = 0;            //表示从栈顶的皇后开始
	while (st.top!=0) {            //栈不空时遍历循环
		k = st.top;
		find = false;
		for (int j = st.col[k] + 1;j <= n;j++)   //寻找合适标号
			if (place(st, k, j)) {               //将其位置（i，j）拿去place中测试
				st.col[st.top] = j;
				find = true;
				break;
			}
			if (find) {                  //如果找到位置
				if (k == n)             //全找到直接输出
					dispasolution(st);  
				else {
					st.top++;          //还没找完先入栈
					st.col[st.top] = 0;//新入栈的皇后从第0列开始重新测试
				}
			}
			else {
				st.top--;                //皇后位置不合适，则回溯。
			}
		}
	}
int main() {
	int n;
	printf("皇后问题（n<20）n=");
	cin >> n;
	if (n > 20) {
		printf("n过大\n");

	}
	else {
		printf("解法如下：\n");
		queen(n);
	}
}