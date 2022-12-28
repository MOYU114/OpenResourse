#include<iostream>
void DBubbleSort(int a[], int n) {
	int i = 0, j;
	int temp;
	bool flag = true;
	while (flag) {
		for (j = n - 1 - i;j > i;j--) {
			if (a[j] < a[j - 1]) //�ɺ���ǰð��СԪ��
			{
				flag = true;
				temp = a[j];a[j] = a[j - 1];a[j - 1] = temp;
			}
		}
		for (j = i;j < n - i - 1;j++) {
			if (a[j] > a[j + 1]) //��ǰ���ð��СԪ��
			{
				flag = true;
				temp = a[j];a[j] = a[j + 1];a[j + 1] = temp;
			}
		}
		if (!flag) return;
		i++;
	}
}
int QuickSelect(int a[], int s, int t, int k) {//��a[s..t]�������ҵ�kС��Ԫ��
	int i = s, j = k;
	int temp;
	if (s < t) {
		temp = a[s];
		while (i != j) {//���������м�ɨ��,ֱ��i==jΪֹ
			while (j > i&& a[i] >= temp)
				i++;//��������ɨ��,�ҵ�1���ؼ���С��tmp��a[j]
			a[i] = a[j];//��a[j]ǰ�Ƶ�a[i]��λ��
			while (j < i&& a[i] <= temp)
				i++;//��������ɨ��,�ҵ�1���ؼ��ִ���tmp��a[i]
			a[j] = a[i];//��a[i]���Ƶ�a[j]��λ��
		}
		a[i] = temp;
		if (k - 1 == i) return a[i];
		else if(k-1<i) return QuickSelect(a, s, i - 1, k);//���������еݹ����
		else return QuickSelect(a, i + 1,t, k);//���������еݹ����
	}
	else if (s == t && s == k - 1) //������ֻ��һ��Ԫ����ΪR[k-1]
		return a[k - 1];
	else
		return -1; //k���󷵻�����ֵ-1
}

int main() {

}