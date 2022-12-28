#include<iostream>
#include<malloc.h>
using namespace std;
typedef struct {
	int key;
	int data;
}RecType;
void disppart(RecType R[], int s, int t) {//显示结果 
	static int i = 1;
	int j;
	printf("第%d次划分", i);
	for (j = 0;j < s;j++)
		printf("  ");
	for (j = s;j <= t;j++)
		printf("%3d", R[j].key);
	printf("\n");
	i++;
}
int partition(RecType R[], int s, int t) {//开始一次划分 
	int i = s, j = t;
	RecType tmp = R[i];
	while (i < j) {
		while (j > i && R[j].key >= tmp.key)
			j--;
		R[i] = R[j];
		while (i < j && R[i].key <= tmp.key)
			i++;
		R[j] = R[i];
	}
	R[i] = tmp;
	disppart(R, s, t);
	return i;
}
void QuickSort(RecType R[], int s, int t) {
	int i;
	if (s < t) {
		i = partition(R, s, t);
		QuickSort(R, s, i - 1);
		QuickSort(R, i + 1, t);
	}
}
int main() {
	int n = 10;
	RecType R[100];
	int a[] = { 6,8,7,9,0,1,3,2,4,5 };
	for (int i = 0;i < n;i++) {
		R[i].key = a[i];
	}
	cout << "排序前： ";
	for (int i = 0;i < n;i++) {
		cout << R[i].key << " ";
	}
	cout << endl;
	QuickSort(R, 0, n - 1);
	cout << "排序后： ";
	for (int i = 0;i < n;i++) {
		cout << R[i].key << " ";
	}
	cout << endl;

}