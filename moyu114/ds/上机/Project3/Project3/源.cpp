#include<iostream>
#include<stdlib.h>
#define MAXSIZE 100
using namespace std;
typedef struct {
	int col[MAXSIZE];         //col[i]Ϊ�ʺ�λ�ñ�ʶ
	int top;                  //ջ��ָ��
}StackType;
void dispasolution(StackType st) {    //���һ����
	static int cnt = 0;
	printf("��%d���⣺",++cnt);
	for (int i = 1;i <= st.top;i++)
		printf("(%d,%d)", i, st.col[i]);
	printf("\n");
}
bool place(StackType st, int k, int j) { //�����������飬���Ƿ��г�ͻ
	int i = 1;
	if (k == 1)return true;             //��һ���ʺ�
	while (i <= k - 1) {                //���Ƿ���ǰ��Ļʺ��г�ͻ
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
	st.top = 0;                     //�ӵ�һ�п�ʼ����ʼ���Ϊ0
	st.top++;
	st.col[st.top] = 0;            //��ʾ��ջ���Ļʺ�ʼ
	while (st.top!=0) {            //ջ����ʱ����ѭ��
		k = st.top;
		find = false;
		for (int j = st.col[k] + 1;j <= n;j++)   //Ѱ�Һ��ʱ��
			if (place(st, k, j)) {               //����λ�ã�i��j����ȥplace�в���
				st.col[st.top] = j;
				find = true;
				break;
			}
			if (find) {                  //����ҵ�λ��
				if (k == n)             //ȫ�ҵ�ֱ�����
					dispasolution(st);  
				else {
					st.top++;          //��û��������ջ
					st.col[st.top] = 0;//����ջ�Ļʺ�ӵ�0�п�ʼ���²���
				}
			}
			else {
				st.top--;                //�ʺ�λ�ò����ʣ�����ݡ�
			}
		}
	}
int main() {
	int n;
	printf("�ʺ����⣨n<20��n=");
	cin >> n;
	if (n > 20) {
		printf("n����\n");

	}
	else {
		printf("�ⷨ���£�\n");
		queen(n);
	}
}