# Guion de la presentación (11 láminas, unos 9 minutos)

Los tiempos de cada lámina son una guía a ritmo normal de habla y no están cronometrados.

## 1. Portada (0:15)

Buenos días. Voy a presentar mi segundo informe, que trata sobre planificación quirúrgica con dPSO y un control aprendido de la perturbación en dos niveles.

## 2. Punto de partida: tres vacíos y cómo los abordamos (1:00)

Al revisar el estado del arte, encontré 3 brechas.

El primero tiene que ver con los bandidos multibrazo. Que en pocas palabras, un bandido es un método que aprende, con recompensas, cuál de unas pocas opciones conviene elegir. Y estos métodos se han usado para decidir qué operador aplicar, pero no para decidir cuánta perturbación aplicar.

El segundo es que, en los enjambres de partículas, el control de parámetros sigue reglas fijas de antemano.

Y el tercero es que no hay una evaluación en programación quirúrgica que separe dos cosas: el efecto de añadir perturbación y el efecto de aprender a elegirla.

Entonces lo que propongo abarca cada uno de esos puntos: un control en dos niveles, una recompensa propia para cada nivel y una ablación con el mismo presupuesto. Entonces, el objetivo es añadir a dPSO un control que aprenda, durante la ejecución, cuánto y cómo perturbar la solución.

## 3. Arquitectura híbrida (1:15)

Aquí se ve el esquema general de la propuesta.

Al centro está dPSO, que mantiene un enjambre de partículas. Cada partícula es un candidato, o sea, un orden de prioridad de los trabajos y una sala para cada operación. A la derecha está el evaluador del solver, que toma el candidato, construye el calendario y devuelve su fitness. Ese fitness combina el fin de la jornada, las esperas entre anestesia y cirugía, y el balance de salas.

Lo que yo agrego son las dos cajas de la izquierda, que son dos bandidos multibrazo. Donde el bandido tiene unas pocas opciones: elige una, observa qué tan bien le fue, y con eso actualiza cuánto vale esa opción. Y con el tiempo va prefiriendo la que mejor le ha resultado.

En mi caso, el bandido de arriba decide la intensidad, que es la probabilidad con que se muta una partícula. El de abajo decide el movimiento, que es el tipo de mutación que se aplica.

Las líneas punteadas son la recompensa que vuelve a cada bandido, y es distinta en cada nivel.

## 4. Nivel 1: intensidad (1:00)

El primer nivel del bandido dice cuánto perturbar. Tiene tres opciones: mutar con probabilidad 0,05, 0,15 o 0,30.

Se elige por bloques de cinco iteraciones. Donde al principio se prueban las tres opciones una vez. Luego se usa una política que se llama épsilon-voraz: que significa que el 90 % de las veces elige la opción que mejor le ha resultado, y el 10 % restante elige una al azar, para no dejar de explorar.

Al terminar cada bloque, se mira cuánto cambió la mejor solución del enjambre, como proporción de lo que valía antes. Esa proporción es la recompensa: si no hubo mejora, vale cero.

Cada opción tiene además un valor Q, que es la estimación de qué tan buena ha sido esa probabilidad hasta ahora. Y con la recompensa del bloque se corrige el Q de la opción que se usó.

Entonces por ejemplo. supongamos que en un bloque el mejor fitness baja de 1000 a 980. Mejoró 20 puntos sobre 1000, o sea, un 2 %, así que la recompensa de ese bloque es 0,02.

Por otro lado, la opción que se usó tenía guardado un Q de 0,01, y lo que acaba de observar es 0,02. La diferencia entre ambos es 0,01. El bandido no reemplaza el valor antiguo por el nuevo, sino que avanza solo un 20% fijo de esa diferencia, que es 0,002. Por eso el Q queda en 0,012.

## 5. Nivel 2: movimiento (0:50)

El segundo nivel dice cómo perturbar. Hay tres movimientos, que están en el dibujo: intercambiar dos trabajos en el orden, sacar un trabajo y ponerlo en otra posición, o cambiar la sala de una operación.

A diferencia del bandido anterior, este nivel decide en cada mutación y no por bloques. Entonces su recompensa también es más simple: vale uno si la partícula mutada mejoró y cero si no. De esa forma, el valor de cada movimiento estima con qué frecuencia ha servido últimamente.

Los dos bandidos son distintos a propósito, porque la intensidad afecta a todo el enjambre, mientras que el movimiento afecta a una sola partícula.

## 6. Siete variantes para tres preguntas (1:00)

Con esto paso a la configuración experimental. Donde lo que hice fue comparar siete variantes de dPSO.

A1 es la dPSO original, sin mutación. A2 muta con una probabilidad fija. A3 elige la probabilidad al azar y A4 la ajusta con reglas simples. Después vienen las que aprenden: A5 aprende solo la intensidad, A6 solo el movimiento, y A7 es la propuesta completa.

Con estas variantes busco responder tres preguntas. La primera es si basta con añadir mutación para que dPSO mejore, sin ningún aprendizaje. La segunda es si vale la pena aprender la intensidad, en lugar de dejarla fija, elegirla al azar o ajustarla con reglas. Y la tercera es si vale la pena aprender también el tipo de movimiento, ya sea por sí solo o junto con la intensidad.

## 7. Instancias (0:40)

En cuanto a las instancias, usaré las 37 que tiene el repositorio, y las dividí en tres conjuntos según para qué se usan.

Cuatro instancias pequeñas sirven solo para ver que esté todo bien implementado.

Ocho son de desarrollo. Son las instancias donde sí puedo probar cosas y tomar decisiones. Con estas podría elegir, por ejemplo, la probabilidad fija que usan las variantes A2 y A6.

Las 25 restantes son de prueba. Son las que se usan para la comparación final, y por eso no las ejecuto hasta que toda la configuración está cerrada.

Además, nueve de las de prueba son más grandes que cualquiera de desarrollo, de 40 a 60 trabajos, así que permiten ver qué pasa cuando aumenta la escala.

## 8. Protocolo: una comparación justa (0:45)

Para que la comparación sea justa, todas las variantes usan el mismo presupuesto, que son 10 000 evaluaciones por corrida. Cada variante se ejecuta 30 veces por instancia con las mismas semillas, de modo que dos variantes con la misma semilla parten del mismo enjambre inicial.

Los parámetros del control los fijé de antemano, con valores habituales en la literatura fijos.

Lo único que elegí con datos fue la probabilidad fija de A2 y A6. Para eso hice un piloto con las instancias de desarrollo, y el valor 0,30 fue el que obtuvo el mejor resultado.

## 9. Medidas y análisis estadístico (0:45)

Por cada corrida registro el fitness, el fin de jornada, las esperas y el tiempo. En las variantes con bandidos guardo además qué acción eligió en cada bloque, porque eso permite ver qué fue lo que aprendió.

Para comparar, la unidad es la instancia, así que voy a tomar la mediana de las 30 corridas. Sobre las 25 instancias de prueba aplico la prueba de Friedman, que indica si hay diferencias entre las siete variantes. Si las hay, comparo la propuesta contra cada una de las otras con Wilcoxon, y corrijo con Holm porque son varias comparaciones.

Junto con cada valor p voy a informar el tamaño de la diferencia y cuántas instancias gana, empata y pierde la propuesta.

## 10. Conclusiones (0:45)

Para cerrar entonces, lo que propongo es un control en dos niveles sobre dPSO, que aprende durante la ejecución cuánto y cómo perturbar, con una recompensa adaptada a cada nivel.

A la fecha tengo la implementación verificada, el piloto terminado y el protocolo cerrado, que se puede ejecutar en menos de una hora.

Lo que falta son los resultados de la comparación final, que voy a presentar en la siguiente entrega. También hay límites que reconocer: las instancias son sintéticas y deterministas, la comparación es interna a dPSO, y la función objetivo agrega varios términos con pesos fijos.

## 11. Cierre (0:05)

Eso sería todo. Muchas gracias.

---

## Respuestas cortas para preguntas probables

- **¿Por qué no hay resultados?** Esta entrega pedía la configuración experimental. La prueba final ya está lista para ejecutarse y toma menos de una hora.
- **¿Por qué dos recompensas distintas?** Porque la intensidad se decide unas 66 veces por corrida y afecta a todo el enjambre, mientras que el movimiento recibe del orden de 1500 observaciones y afecta a una sola partícula. (Anexo: "Dos niveles, dos señales".)
- **¿Por qué 10 000 evaluaciones?** Son 30 de la población inicial, 66 bloques de 150 y un bloque final truncado de 70. (Anexo: "Anatomía de una corrida".)
- **¿Qué mide exactamente la función objetivo?** (Anexo: "Representación del candidato y función de evaluación".)
- **¿Qué mostró el piloto?** Además de elegir 0,30, sirvió para verificar la implementación: cada corrida consume exactamente 10 000 evaluaciones y las 1440 corridas fueron factibles. También mostró que el límite de espera de cada instancia no actúa como restricción, porque el evaluador usa una cota global mayor; por eso voy a informar la espera máxima. (Anexo: "Piloto en desarrollo y verificación previa".)
- **¿Qué viene después?** Darle contexto al control, comparar con otras metaheurísticas del repositorio y probar con duraciones inciertas. (Anexo: "Limitaciones y trabajo futuro".)
