
Ctrl + Shift + V

### Берём производную по вектору бетта 

$$
\boxed{L(\boldsymbol{\beta})=\mathbf y^T\mathbf y-\mathbf y^TX\boldsymbol{\beta}-\boldsymbol{\beta}^TX^T\mathbf y+\boldsymbol{\beta}TX^TX\boldsymbol{\beta}}
$$

Функция зависит сразу от нескольких переменных:

$$
\beta_0,\beta_1,\dots,\beta_p
$$

Поэтому производная по вектору $\boldsymbol{\beta}$ — это вектор частных производных:

$$
\frac{\partial L}{\partial\boldsymbol{\beta}}
=
\begin{pmatrix}
\frac{\partial L}{\partial\beta_0}\\
\frac{\partial L}{\partial\beta_1}\\
\vdots\\
\frac{\partial L}{\partial\beta_p}
\end{pmatrix}.
$$

Рассмотрим каждое слагаемое функции $L$ отдельно.

---

## 1. Производная $\mathbf y^T\mathbf y$

Рассмотрим:

$$
\mathbf y^T\mathbf y.
$$

В этом выражении нет $\boldsymbol{\beta}$.

Следовательно, относительно $\boldsymbol{\beta}$ это константа.

Производная константы равна нулю:

$$
\boxed{
\frac{\partial}{\partial\boldsymbol{\beta}}
(\mathbf y^T\mathbf y)
=
0
}
$$

---

## 2. Производная $-\mathbf y^TX\boldsymbol{\beta}$

Рассмотрим:

$$
-\mathbf y^TX\boldsymbol{\beta}.
$$

Для простоты обозначим:

$$
X^T\mathbf y=\mathbf a.
$$

Тогда:

$$
\mathbf y^TX\boldsymbol{\beta}
=
\mathbf a^T\boldsymbol{\beta}.
$$

Пусть:

$$
\mathbf a=
\begin{pmatrix}
a_0\\
a_1\\
\vdots\\
a_p
\end{pmatrix},
\qquad
\boldsymbol{\beta}
=
\begin{pmatrix}
\beta_0\\
\beta_1\\
\vdots\\
\beta_p
\end{pmatrix}.
$$

Тогда:

$$
\mathbf a^T\boldsymbol{\beta}
=
\begin{pmatrix}
a_0&a_1&\dots&a_p
\end{pmatrix}
\begin{pmatrix}
\beta_0\\
\beta_1\\
\vdots\\
\beta_p
\end{pmatrix}.
$$

По правилу умножения строки на столбец:

$$
\mathbf a^T\boldsymbol{\beta}
=
a_0\beta_0+a_1\beta_1+\dots+a_p\beta_p.
$$

Теперь берём частные производные.

По $\beta_0$:

$$
\frac{\partial}{\partial\beta_0}
(a_0\beta_0+a_1\beta_1+\dots+a_p\beta_p)
=
a_0.
$$

По $\beta_1$:

$$
\frac{\partial}{\partial\beta_1}
(a_0\beta_0+a_1\beta_1+\dots+a_p\beta_p)
=
a_1.
$$

И так далее.

Поэтому:

$$
\frac{\partial}{\partial\boldsymbol{\beta}}
(\mathbf a^T\boldsymbol{\beta})
=
\mathbf a.
$$

Так как:

$$
\mathbf a=X^T\mathbf y,
$$

получаем:

$$
\frac{\partial}{\partial\boldsymbol{\beta}}
(\mathbf y^TX\boldsymbol{\beta})
=
X^T\mathbf y.
$$

Перед выражением стоит минус, поэтому:

$$
\boxed{
\frac{\partial}{\partial\boldsymbol{\beta}}
(-\mathbf y^TX\boldsymbol{\beta})
=
-X^T\mathbf y
}
$$

---

## 3. Производная $-\boldsymbol{\beta}^TX^T\mathbf y$

Рассмотрим:

$$
-\boldsymbol{\beta}^TX^T\mathbf y.
$$

Заметим, что:

$$
\mathbf y^TX\boldsymbol{\beta}
=
\boldsymbol{\beta}^TX^T\mathbf y.
$$

Действительно:

$$
\mathbf y^TX\boldsymbol{\beta}
=
(\mathbf y^TX\boldsymbol{\beta})^T.
$$

Используя правило:

$$
(ABC)^T=C^TB^TA^T,
$$

получаем:

$$
(\mathbf y^TX\boldsymbol{\beta})^T
=
\boldsymbol{\beta}^TX^T\mathbf y.
$$

Следовательно:

$$
\mathbf y^TX\boldsymbol{\beta}
=
\boldsymbol{\beta}^TX^T\mathbf y.
$$

Это одно и то же число.

Поэтому производная аналогична предыдущему случаю:

$$
\boxed{
\frac{\partial}{\partial\boldsymbol{\beta}}
(-\boldsymbol{\beta}^TX^T\mathbf y)
=
-X^T\mathbf y
}
$$

Таким образом, два средних слагаемых вместе дают:

$$
-X^T\mathbf y-X^T\mathbf y
=
-2X^T\mathbf y.
$$

---

# 4. Производная $\boldsymbol{\beta}^TX^TX\boldsymbol{\beta}$

Теперь рассмотрим наиболее сложное слагаемое:

$$
\boldsymbol{\beta}^TX^TX\boldsymbol{\beta}.
$$

Для начала обозначим:

$$
A=X^TX.
$$

Тогда выражение принимает вид:

$$
\boldsymbol{\beta}^TA\boldsymbol{\beta}.
$$

Нужно найти:

$$
\frac{\partial}{\partial\boldsymbol{\beta}}
(\boldsymbol{\beta}^TA\boldsymbol{\beta}).
$$

Чтобы понять эту формулу, сначала рассмотрим простой случай с двумя коэффициентами.

Пусть:

$$
\boldsymbol{\beta}
=
\begin{pmatrix}
\beta_1\\
\beta_2
\end{pmatrix},
\qquad
A=
\begin{pmatrix}
a&b\\
c&d
\end{pmatrix}.
$$

Тогда:

$$
\boldsymbol{\beta}^TA\boldsymbol{\beta}
=
\begin{pmatrix}
\beta_1&\beta_2
\end{pmatrix}
\begin{pmatrix}
a&b\\
c&d
\end{pmatrix}
\begin{pmatrix}
\beta_1\\
\beta_2
\end{pmatrix}.
$$

Сначала перемножаем первые две матрицы:

$$
\begin{pmatrix}
\beta_1&\beta_2
\end{pmatrix}
\begin{pmatrix}
a&b\\
c&d
\end{pmatrix}
=
\begin{pmatrix}
a\beta_1+c\beta_2&
b\beta_1+d\beta_2
\end{pmatrix}.
$$

Теперь умножаем на последний столбец:

$$
\begin{pmatrix}
a\beta_1+c\beta_2&
b\beta_1+d\beta_2
\end{pmatrix}
\begin{pmatrix}
\beta_1\\
\beta_2
\end{pmatrix}.
$$

Получаем:

$$
a\beta_1^2
+c\beta_1\beta_2
+b\beta_1\beta_2
+d\beta_2^2.
$$

Собираем подобные слагаемые:

$$
\boxed{
\boldsymbol{\beta}^TA\boldsymbol{\beta}
=
a\beta_1^2
+(b+c)\beta_1\beta_2
+d\beta_2^2
}
$$

Теперь это обычная функция двух переменных.

---

## 4.1. Берём обычные частные производные

По $\beta_1$:

$$
\frac{\partial}{\partial\beta_1}
\left(
a\beta_1^2
+(b+c)\beta_1\beta_2
+d\beta_2^2
\right)
=
2a\beta_1+(b+c)\beta_2.
$$

По $\beta_2$:

$$
\frac{\partial}{\partial\beta_2}
\left(
a\beta_1^2
+(b+c)\beta_1\beta_2
+d\beta_2^2
\right)
=
(b+c)\beta_1+2d\beta_2.
$$

Следовательно, вектор частных производных:

$$
\boxed{
\frac{\partial}{\partial\boldsymbol{\beta}}
(\boldsymbol{\beta}^TA\boldsymbol{\beta})
=
\begin{pmatrix}
2a\beta_1+(b+c)\beta_2\\
(b+c)\beta_1+2d\beta_2
\end{pmatrix}
}
$$

 В матричной форме это:  $$ \boxed{ \frac{\partial}{\partial\boldsymbol{\beta}} (\boldsymbol{\beta}^TA\boldsymbol{\beta})  =  (A+A^T)\boldsymbol{\beta}  }  $$
> [!success]- почему $(A+A^T)\boldsymbol{\beta}$?
>
>
> Имеем:
>
> $$
> A=
> \begin{pmatrix}
> a&b\\
> c&d
> \end{pmatrix}.
> $$
>
> Транспонируем:
>
> $$
> A^T=
> \begin{pmatrix}
> a&c\\
> b&d
> \end{pmatrix}.
> $$
>
> Складываем:
>
> $$
> A+A^T
> =
> \begin{pmatrix}
> a&b\\
> c&d
> \end{pmatrix}
> +
> \begin{pmatrix}
> a&c\\
> b&d
> \end{pmatrix}.
> $$
>
> Получаем:
>
> $$
> A+A^T
> =
> \begin{pmatrix}
> 2a&b+c\\
> b+c&2d
> \end{pmatrix}.
> $$
>
> Теперь умножаем на $\boldsymbol{\beta}$:
>
> $$
> (A+A^T)\boldsymbol{\beta}
> =
> \begin{pmatrix}
> 2a&b+c\\
> b+c&2d
> \end{pmatrix}
> \begin{pmatrix}
> \beta_1\\
> \beta_2
> \end{pmatrix}.
> $$
>
> Перемножаем матрицу и вектор построчно:
>
> $$
> (A+A^T)\boldsymbol{\beta}
> =
> \begin{pmatrix}
> 2a\beta_1+(b+c)\beta_2\\
> (b+c)\beta_1+2d\beta_2
> \end{pmatrix}.
> $$
>
> это **ровно тот же самый вектор**, который мы получили обычными частными производными:
>
> $$
> \boxed{
> \frac{\partial}{\partial\boldsymbol{\beta}}
> (\boldsymbol{\beta}^TA\boldsymbol{\beta})
> =
> (A+A^T)\boldsymbol{\beta}
> }
> $$
>
> То есть:
>
> $$
> \boxed{
> (A+A^T)\boldsymbol{\beta}
> }
> $$
>
> **т.е это компактная матричная запись уже найденного вектора частных производных**.

---
# 4.2. Возвращаемся к $A=X^TX$

Мы ввели:

$$
A=X^TX.
$$

Поэтому:

$$
\frac{\partial}{\partial\boldsymbol{\beta}}
(\boldsymbol{\beta}^TX^TX\boldsymbol{\beta})
=
\left(
X^TX+(X^TX)^T
\right)\boldsymbol{\beta}.
$$

Теперь используем свойство транспонирования произведения:

$$
(X^TX)^T
=
X^T(X^T)^T.
$$

Так как:

$$
(X^T)^T=X,
$$

получаем:

$$
(X^TX)^T=X^TX.
$$

То есть матрица $X^TX$ симметрична.

Следовательно:

$$
X^TX+(X^TX)^T
=
X^TX+X^TX
=
2X^TX.
$$

Поэтому:

$$
\boxed{
\frac{\partial}{\partial\boldsymbol{\beta}}
(\boldsymbol{\beta}^TX^TX\boldsymbol{\beta})
=
2X^TX\boldsymbol{\beta}
}
$$

---

# 5. Собираем производные всех слагаемых

Исходная функция:

$$
L(\boldsymbol{\beta})
=
\mathbf y^T\mathbf y
-\mathbf y^TX\boldsymbol{\beta}
-\boldsymbol{\beta}^TX^T\mathbf y
+\boldsymbol{\beta}^TX^TX\boldsymbol{\beta}.
$$

Производная первого слагаемого:

$$
\frac{\partial}{\partial\boldsymbol{\beta}}
(\mathbf y^T\mathbf y)
=
0.
$$

Производная второго:

$$
\frac{\partial}{\partial\boldsymbol{\beta}}
(-\mathbf y^TX\boldsymbol{\beta})
=
-X^T\mathbf y.
$$

Производная третьего:

$$
\frac{\partial}{\partial\boldsymbol{\beta}}
(-\boldsymbol{\beta}^TX^T\mathbf y)
=
-X^T\mathbf y.
$$

Производная четвёртого:

$$
\frac{\partial}{\partial\boldsymbol{\beta}}
(\boldsymbol{\beta}^TX^TX\boldsymbol{\beta})
=
2X^TX\boldsymbol{\beta}.
$$

Складываем:

$$
\frac{\partial L}{\partial\boldsymbol{\beta}}
=
0
-X^T\mathbf y
-X^T\mathbf y
+
2X^TX\boldsymbol{\beta}.
$$

Получаем:

$$
\boxed{
\frac{\partial L}{\partial\boldsymbol{\beta}}
=
-2X^T\mathbf y
+
2X^TX\boldsymbol{\beta}
}
$$


