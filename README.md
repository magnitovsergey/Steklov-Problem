# Спектральная задача Стеклова
## Описание интерфейса
При запуске файла main.py открывается окно с вводом данных. Необходимо ввести:
1. Количество узлов сетки по оси X (целое положительное число);
2. Количество узлов сетки по оси Y (целое положительной число);
3. Шаг интегрирования (вещественное положительное число);
4. Точность полученных вычисление (целое положительное число);
5. Количество желаемых функций.

Вместо пунктов 1 и 2 можно загрузить свою карту области в бинарном виде. Общая форма карты должна представлять из себя прямоугольник, внутри которого должна находиться замкнутая область из единиц (в остальных местах области, то есть там, где задача не определена, должны стоять нули). При успешной загрузки карты соответсвующая кнопка загорится зеленым. Если кнопка "Загрузить файл" меняет цвет на синий, это значит, что карта удалена, и ее нужно загрузить заново. 

После нажатия кнопки "Обработать данные" начинается процесс вычисление, в течение которого кнопка меняет цвет на зеленый и показывает объявление "Обработка данных". Как только кнопка принимает свой начальный фиолетовый цвет, можно смотреть результаты.

## Получение результатов
Помимо встроенного окна с выводом результатов на экран, численные результаты со всем иллюстративным материалом автоматически сохраняются в Results в корневой папке.

## Технические требования
Для запуска представленного приложения необходимо убедиться, что у вас установлены все необходимые библиотеки:
* PyQt5;
* PIL (Python Imaging Library);
* NumPy;
* Matplotlib;
* SciPy;
* os (стандартная библиотека Python для работы с операционной системой);
* joblib.

Также перед началом работы убедитесь, что у вас установлен Python версии 3.6 или выше. 
