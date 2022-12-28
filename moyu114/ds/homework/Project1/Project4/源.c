#include<stdio.h>
void fun(int a[],int n) 
{
	int i=0,j=n-1, temp;
	while (i < j) 
	{
		while(a[i]<0){
			i++;
		}
		while (a[j] >= 0) {
			j--;
		}
		if (i < j) {
			temp = a[i];a[i] = a[j];a[j] = temp;
		}
	}
}
int main() {
	int a[10] = { 1,2,-8,-9,7,6,7,-8,-7,-5 };
	fun(a, 10);
	for (int i = 0;i < 10;i++)
		printf("%d ", a[i]);
}