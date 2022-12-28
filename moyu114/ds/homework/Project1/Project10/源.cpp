#include<iostream>
void DBubbleSort(int a[], int n) {
	int i = 0, j;
	int temp;
	bool flag = true;
	while (flag) {
		for (j = n - 1 - i;j > i;j--) {
			if (a[j] < a[j - 1]) //由后向前冒泡小元素
			{
				flag = true;
				temp = a[j];a[j] = a[j - 1];a[j - 1] = temp;
			}
		}
		for (j = i;j < n - i - 1;j++) {
			if (a[j] > a[j + 1]) //由前向后冒泡小元素
			{
				flag = true;
				temp = a[j];a[j] = a[j + 1];a[j + 1] = temp;
			}
		}
		if (!flag) return;
		i++;
	}
}
int QuickSelect(int a[], int s, int t, int k) {//在a[s..t]序列中找第k小的元素
	int i = s, j = k;
	int temp;
	if (s < t) {
		temp = a[s];
		while (i != j) {//从两端向中间扫描,直至i==j为止
			while (j > i&& a[i] >= temp)
				i++;//从右向左扫描,找第1个关键字小于tmp的a[j]
			a[i] = a[j];//将a[j]前移到a[i]的位置
			while (j < i&& a[i] <= temp)
				i++;//从左向右扫描,找第1个关键字大于tmp的a[i]
			a[j] = a[i];//将a[i]后移到a[j]的位置
		}
		a[i] = temp;
		if (k - 1 == i) return a[i];
		else if(k-1<i) return QuickSelect(a, s, i - 1, k);//在左区间中递归查找
		else return QuickSelect(a, i + 1,t, k);//在右区间中递归查找
	}
	else if (s == t && s == k - 1) //区间内只有一个元素且为R[k-1]
		return a[k - 1];
	else
		return -1; //k错误返回特殊值-1
}

int main() {

}