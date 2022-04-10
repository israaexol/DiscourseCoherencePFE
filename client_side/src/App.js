import './App.css';
import {Accueil, Apropos, Navbar} from './Component';
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

const App = () => {
  return (
    <Router>
        <Navbar/>
        <Routes>
            <Route path='/' element={<Accueil/>}/>
            
            <Route path='/accueil' element={<Accueil/>}/>

            <Route path='/apropos' element={<Apropos/>}/>
        </Routes>
    </Router>      
  );
}

export default App;
