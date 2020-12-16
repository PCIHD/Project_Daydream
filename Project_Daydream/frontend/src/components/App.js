import React , {Component} from 'react';
import {render} from "react-dom";



function App() {
  return (
    <div className="App">
      <h1>Let's Start DayDreaming</h1>
      <Draw />
    </div>
  );
}




export default App;
const container=document.getElementById("app");
render(<App/>,container);