#include"sqlist.h"
typedef int KeyType;
int BinSearch(SqList* S, KeyType k) {
	int low=0, high=S->length-1, mid,cnt=0;
	while (low <= high) {
		mid = (low + high) / 2;
		cnt++;
		printf("第%d比较：在S[%d,%d]中查找元素S[%d]",cnt,low,high,mid);
		if (S->data[mid] == k) {
			cout << "已查询到!" << endl;
			return mid + 1;
		}else if (S->data[mid] >k) {
			cout << "比查找元素大" << endl;
			high = mid - 1;
		}
		else {
			cout << "比查找元素小" << endl;
			low = mid + 1;
		}
	}
}
int main() {
	SqList* s;
	KeyType k = 9;
	int result = -1;
	ElemType a[] = { 1,2,3,4,5,6,7,8,9 };
	InitList(s);
	CreateList(s,a,9);
	 result = BinSearch(s,k);
	 if (result != -1) {
		 printf("查找元素%d在顺序表的%d处\n", k, result);
	}
}