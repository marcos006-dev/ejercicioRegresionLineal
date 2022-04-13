import * as tf from '@tensorflow/tfjs';
const inputValorCalulcar = document.getElementById('valorACalcular');
const contenedorResultado = document.getElementById('resultado');

let modeloEntrenado;

const calcularValoresY = (paramValoresX) => {
  const arrayResultadosY = [];
  for (let i = 0; i < paramValoresX.length; i++) {
    const y = 3 * paramValoresX[i] + 2;
    arrayResultadosY.push(y);
  }
  return arrayResultadosY;
};

const funcionLineal = async () => {
  contenedorResultado.innerHTML = 'El modelo se esta entrenando...';

  const model = tf.sequential();

  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  const valoresInicialesX = [-1, 0, 1, 2, 3, 4];

  const resultadoY = calcularValoresY(valoresInicialesX);

  const xs = tf.tensor2d(valoresInicialesX, [6, 1]);
  const ys = tf.tensor2d(resultadoY, [6, 1]);

  await model.fit(xs, ys, { epochs: 500 });

  inputValorCalulcar.disabled = false;
  inputValorCalulcar.focus();

  modeloEntrenado = model;
  contenedorResultado.innerHTML = 'Modelo entrenado, listo para usar';
};

window.addEventListener('DOMContentLoaded', (event) => {
  funcionLineal();

  inputValorCalulcar.addEventListener('keyup', (event) => {
    if (event.keyCode === 13) {
      event.preventDefault();
      const valorACalcular = parseInt(inputValorCalulcar.value);
      const resultado = modeloEntrenado.predict(
        tf.tensor2d([valorACalcular], [1, 1])
      );
      // console.log(resultado);
      const valorResultado = resultado.dataSync();

      contenedorResultado.innerHTML = `El resultado aproximado para Y es de: ${valorResultado}`;
    }
  });
});
