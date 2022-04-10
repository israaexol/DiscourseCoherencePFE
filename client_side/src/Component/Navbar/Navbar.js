import React from 'react';
import './Navbar.css'
import logo from '../../assets/logo.png'


const Menu =() =>(
    <>
    <p className='link'><a href='#accueil'>Accueil</a></p>
    <p className='link'><a href='#apropos'>À propos</a></p>
    <p className='link'><a href='#contact'>Contactez-nous</a></p>
    </>
)
const Navbar = () => {
  return (
    <div id='navbar'>
        <style>
             @import url("https://fonts.googleapis.com/css2?family=Pacifico"); 
             @import url('https://fonts.googleapis.com/css2?family=Roboto');
        </style>
        <div className='navbar_links'>
            <div className='navbar_logo'>
                <img src={logo} alt="logo"/>
                La coherencia
            </div>
            <div className='navbar_menu'>
                <Menu />
            </div>
        </div>

    </div>
  )
}

export default Navbar