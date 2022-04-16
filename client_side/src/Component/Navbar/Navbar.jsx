import React from 'react';
import './Navbar.css'
import logo from '../../assets/logo.png'
import coherencia from '../../assets/coherencia.png'


const Menu = () => (
    <>
        <div className='menu_div'> < p className='link' > <a href='accueil' > Accueil </a></p > </div>
        <div className='menu_div'> < p  > <a href='apropos' > À propos </a></p > </div>
        <div className='menu_div'> < p > <a href='contact' > Contactez nous </a></p> </div>
    </>
)
const Navbar = () => {
    return (
        <div id='navbar' >
            <div className='navbar_links' >
                < div className='navbar_logo' >
                    <img src={coherencia}
                        alt="logo" />
                    La coherencia
                </div>
                <div className='navbar_menu' >
                    <Menu />
                </div>
            </div>
        </div>
    )
}

export default Navbar